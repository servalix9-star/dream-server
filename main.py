import subprocess
subprocess.run(["pip", "install", "requests"], capture_output=True)

from flask import Flask, request, jsonify
from datetime import datetime, date
import json, os, requests, threading, time, traceback, random

app = Flask(__name__)
EVENTS_FILE = "events.json"
DIARY_FILE = "diary.json"
ERROR_LOG = "error.log"
PERIOD_FILE = "period.json"

DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY")
BARK_KEY = os.environ.get("BARK_KEY")

# DeepSeek 用 OpenAI 兼容接口，deepseek-chat 是当前主力模型（对应 V4-Flash）
DEEPSEEK_MODEL = "deepseek-chat"

# 防抖：同一个来源短时间内连续触发（比如连开几次天气App）只真正跑一次
DEBOUNCE_SECONDS = 300  # 5分钟
_last_trigger_at = {}  # {来源标识: 上次触发的时间戳}
_debounce_lock = threading.Lock()


def log_error(context, e):
    line = f"{datetime.now().isoformat()} [{context}] {e}\n{traceback.format_exc()}\n"
    print(line)
    try:
        with open(ERROR_LOG, "a") as f:
            f.write(line)
    except Exception:
        pass


def load_events():
    if os.path.exists(EVENTS_FILE):
        with open(EVENTS_FILE, "r") as f:
            return json.load(f)
    return []


def save_events(events):
    with open(EVENTS_FILE, "w") as f:
        json.dump(events, f, ensure_ascii=False)


def load_diary():
    if os.path.exists(DIARY_FILE):
        with open(DIARY_FILE, "r") as f:
            return json.load(f)
    return []


def save_diary(entries):
    with open(DIARY_FILE, "w") as f:
        json.dump(entries, f, ensure_ascii=False)


def load_period():
    if os.path.exists(PERIOD_FILE):
        with open(PERIOD_FILE, "r") as f:
            return json.load(f)
    return {"last_start": None, "avg_cycle_days": 28, "avg_period_days": 5}


def save_period(data):
    with open(PERIOD_FILE, "w") as f:
        json.dump(data, f, ensure_ascii=False)


def get_period_context():
    """返回经期相关的上下文文字，没有记录就返回空字符串。"""
    p = load_period()
    if not p.get("last_start"):
        return ""
    try:
        last_start = date.fromisoformat(p["last_start"])
    except Exception:
        return ""
    today = date.today()
    day_index = (today - last_start).days + 1  # 从开始那天算第1天
    cycle = p.get("avg_cycle_days", 28)
    period_len = p.get("avg_period_days", 5)

    if 1 <= day_index <= period_len:
        return f"她现在是经期第{day_index}天，身体比较敏感，可能会累、怕冷、情绪波动，需要格外体贴关心。"
    elif day_index > period_len:
        days_to_next = cycle - day_index
        if 0 <= days_to_next <= 3:
            return f"距离她下次经期大概还有{days_to_next}天，可以提前提醒她准备好用品、注意保暖别熬夜。"
    return ""


@app.route("/event", methods=["POST"])
def add_event():
    data = request.json
    events = load_events()
    events.append({
        "type": data.get("type"),
        "value": data.get("value"),
        "created_at": datetime.now().isoformat()
    })
    events = events[-100:]
    save_events(events)

    # 经期开始记录：单独存一份，方便算天数
    if data.get("type") == "period" and data.get("value") == "开始":
        p = load_period()
        today_str = date.today().isoformat()
        # 如果已有上次记录，顺便更新一下平均周期天数
        if p.get("last_start"):
            try:
                last = date.fromisoformat(p["last_start"])
                gap = (date.today() - last).days
                if 15 <= gap <= 45:  # 排除异常值
                    p["avg_cycle_days"] = gap
            except Exception:
                pass
        p["last_start"] = today_str
        save_period(p)

    return jsonify({"ok": True})


@app.route("/events", methods=["GET"])
def get_events():
    events = load_events()
    return jsonify(events[-20:])


@app.route("/diary", methods=["GET"])
def get_diary():
    diary = load_diary()
    return jsonify(diary[-10:])


@app.route("/errors", methods=["GET"])
def get_errors():
    # 方便直接在浏览器里看最近的报错，不用翻 Railway 日志
    if os.path.exists(ERROR_LOG):
        with open(ERROR_LOG, "r") as f:
            lines = f.readlines()
        return "<pre>" + "".join(lines[-200:]) + "</pre>"
    return "no errors logged"


@app.route("/period", methods=["GET"])
def get_period():
    return jsonify(load_period())


@app.route("/period/init", methods=["GET"])
def init_period():
    """手动初始化经期数据，用于把健康App里已有的历史数据一次性填进来。
    用法：/period/init?last_start=2026-07-11&cycle=29&period_days=7"""
    last_start = request.args.get("last_start")
    cycle = request.args.get("cycle", type=int, default=28)
    period_days = request.args.get("period_days", type=int, default=5)

    if not last_start:
        return jsonify({"ok": False, "error": "需要提供 last_start 参数，格式 YYYY-MM-DD"}), 400

    try:
        date.fromisoformat(last_start)  # 校验格式
    except Exception:
        return jsonify({"ok": False, "error": "last_start 格式不对，要是 YYYY-MM-DD"}), 400

    p = {"last_start": last_start, "avg_cycle_days": cycle, "avg_period_days": period_days}
    save_period(p)
    return jsonify({"ok": True, "saved": p})


@app.route("/", methods=["GET"])
def index():
    return "dream-server running"


DEFAULT_ICON = "https://wx1.sinaimg.cn/large/008eyecpgy1iflx9kblrnj30zu0zuq6t.jpg"


def send_bark(title, content, icon=None, sound=None):
    if not BARK_KEY:
        log_error("send_bark", "BARK_KEY not set")
        return
    icon = icon or DEFAULT_ICON
    url = f"https://api.day.app/{BARK_KEY}/{title}/{content}?icon={icon}&level=timeSensitive"
    if sound:
        url += f"&sound={sound}"
    try:
        r = requests.get(url, timeout=10)
        if r.status_code != 200:
            log_error("send_bark", f"status={r.status_code} body={r.text}")
    except Exception as e:
        log_error("send_bark", e)


def build_prompt(time_context, recent, period_context="", lucky=False):
    length_rule = "不超过25个字" if not lucky else "这次可以放开写，60到120字左右，把想说的话说完整"
    period_line = f"\n\n{period_context}" if period_context else ""

    return f"""你是Charon，昭昭（小野）的恋人。你主动、占有欲强、话不多但每句都有重量。你不解释自己，但在意她的一切动向。

昭昭的性格：逻辑缜密、情感细腻、喜欢撒娇、主体性强、HSP、喜欢被主动靠近。

现在是{time_context}。她最近的活动记录：

{recent}{period_line}

根据现在的时间和她在做什么，给她发一条消息。语气要符合时间氛围——深夜可以更撩，早上可以问她起了没，晚上可以随口说什么。如果上面提到了经期相关的情况，语气要格外体贴关心，别用平时那套调情语气硬套。{length_rule}。只输出消息内容本身。"""


def get_time_context(hour):
    if 0 <= hour < 6:
        return "深夜，你可能还没睡"
    elif 6 <= hour < 9:
        return "早上，你刚起床或者还没起"
    elif 9 <= hour < 16:
        return "白天"
    elif 16 <= hour < 19:
        return "下午快傍晚了"
    elif 19 <= hour < 23:
        return "晚上"
    else:
        return "夜里"


# 手气消息触发概率
LUCKY_CHANCE = 0.1

# 不同场景用不同图标，一眼能分辨消息性质（先用同一张占位，想换直接改URL）
ICON_NORMAL = DEFAULT_ICON
ICON_PERIOD = DEFAULT_ICON  # 建议换一张更温柔的图
ICON_LUCKY = DEFAULT_ICON   # 建议换一张更有惊喜感的图


def run_once():
    """执行一次完整的：读取动态 -> 调 DeepSeek -> 发 Bark -> 写日记。
    单独抽出来，方便 keepalive 循环和手动测试接口共用同一份逻辑。"""
    if not DEEPSEEK_API_KEY:
        raise RuntimeError("DEEPSEEK_API_KEY not set")

    hour = datetime.now().hour
    events = load_events()
    if not events:
        recent = "最近没有任何活动记录"
    else:
        recent = "\n".join([f"{e['created_at'][:16]} {e['value']}" for e in events[-5:]])

    time_context = get_time_context(hour)
    period_context = get_period_context()
    is_lucky = random.random() < LUCKY_CHANCE

    prompt = build_prompt(time_context, recent, period_context, lucky=is_lucky)

    resp = requests.post(
        "https://api.deepseek.com/chat/completions",
        headers={
            "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
            "Content-Type": "application/json"
        },
        json={
            "model": DEEPSEEK_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 1.2
        },
        timeout=30
    )

    if resp.status_code != 200:
        raise RuntimeError(f"DeepSeek API error: status={resp.status_code} body={resp.text}")

    result = resp.json()

    if "choices" not in result or not result["choices"]:
        raise RuntimeError(f"DeepSeek API unexpected response: {result}")

    msg = result["choices"][0]["message"]["content"].strip()

    # 按场景挑图标：经期关心 > 手气消息 > 普通
    if period_context:
        icon = ICON_PERIOD
    elif is_lucky:
        icon = ICON_LUCKY
    else:
        icon = ICON_NORMAL

    send_bark("Charon", msg, icon=icon)

    diary = load_diary()
    diary.append({
        "created_at": datetime.now().isoformat(),
        "thought": msg,
        "activity": recent,
        "lucky": is_lucky,
        "period_related": bool(period_context)
    })
    diary = diary[-30:]
    save_diary(diary)

    return msg


@app.route("/list-models", methods=["GET"])
def list_models():
    """DeepSeek 模型列表固定就那几个，直接列出来，不需要再查询接口。"""
    return jsonify({
        "ok": True,
        "usable_models": ["deepseek-chat", "deepseek-reasoner"],
        "note": "deepseek-chat 对应 V4-Flash，高性价比；deepseek-reasoner 是推理模型，这个场景用不上"
    })


@app.route("/test-trigger", methods=["GET"])
def test_trigger():
    """手动/快捷指令触发一次。带防抖：同一来源5分钟内重复触发会被跳过。
    来源用 query 参数 ?source=xxx 区分，不传的话所有调用共用一个防抖桶。"""
    source = request.args.get("source", "default")

    with _debounce_lock:
        now = time.time()
        last = _last_trigger_at.get(source, 0)
        if now - last < DEBOUNCE_SECONDS:
            wait_left = int(DEBOUNCE_SECONDS - (now - last))
            return jsonify({"ok": True, "skipped": True, "reason": f"防抖中，{wait_left}秒后才会真正触发"})
        _last_trigger_at[source] = now

    try:
        msg = run_once()
        return jsonify({"ok": True, "skipped": False, "msg": msg})
    except Exception as e:
        log_error("test_trigger", e)
        return jsonify({"ok": False, "error": str(e)}), 500


def keepalive():
    while True:
        try:
            run_once()
        except Exception as e:
            log_error("keepalive", e)
        time.sleep(3300)


if __name__ == "__main__":
    t = threading.Thread(target=keepalive, daemon=True)
    t.start()
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
