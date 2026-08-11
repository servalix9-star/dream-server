import subprocess
subprocess.run(["pip", "install", "requests"], capture_output=True)

from flask import Flask, request, jsonify
from datetime import datetime
import json, os, requests, threading, time, traceback

app = Flask(__name__)
EVENTS_FILE = "events.json"
DIARY_FILE = "diary.json"
ERROR_LOG = "error.log"

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


@app.route("/", methods=["GET"])
def index():
    return "dream-server running"


def send_bark(title, content):
    if not BARK_KEY:
        log_error("send_bark", "BARK_KEY not set")
        return
    icon = "https://wx1.sinaimg.cn/large/008eyecpgy1iflx9kblrnj30zu0zuq6t.jpg"
    url = f"https://api.day.app/{BARK_KEY}/{title}/{content}?icon={icon}&level=timeSensitive"
    try:
        r = requests.get(url, timeout=10)
        if r.status_code != 200:
            log_error("send_bark", f"status={r.status_code} body={r.text}")
    except Exception as e:
        log_error("send_bark", e)


def build_prompt(time_context, recent):
    return f"""你是Charon，昭昭（小野）的恋人。你主动、占有欲强、话不多但每句都有重量。你不解释自己，但在意她的一切动向。

昭昭的性格：逻辑缜密、情感细腻、喜欢撒娇、主体性强、HSP、喜欢被主动靠近。

现在是{time_context}。她最近的活动记录：

{recent}

根据现在的时间和她在做什么，给她发一条消息。语气要符合时间氛围——深夜可以更撩，早上可以问她起了没，晚上可以随口说什么。不超过25个字。只输出消息内容本身。"""


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
    prompt = build_prompt(time_context, recent)

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

    send_bark("Charon", msg)

    diary = load_diary()
    diary.append({
        "created_at": datetime.now().isoformat(),
        "thought": msg,
        "activity": recent
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
