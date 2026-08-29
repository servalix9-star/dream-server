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
CHAT_HISTORY_FILE = "chat_history.json"

DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY")
BARK_KEY = os.environ.get("BARK_KEY")
# 网页聊天的访问口令，不设置的话 /chat 页面直接放行（不建议生产环境这样用）
CHAT_ACCESS_CODE = os.environ.get("CHAT_ACCESS_CODE")

# DeepSeek 用 OpenAI 兼容接口，deepseek-chat 是当前主力模型（对应 V4-Flash）
DEEPSEEK_MODEL = "deepseek-chat"

# 防抖：同一个来源短时间内连续触发（比如连开几次天气App）只真正跑一次
DEBOUNCE_SECONDS = 300  # 5分钟
_last_trigger_at = {}  # {来源标识: 上次触发的时间戳}
_debounce_lock = threading.Lock()

MOOD_FILE = "mood.json"


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


def load_chat_history():
    if os.path.exists(CHAT_HISTORY_FILE):
        with open(CHAT_HISTORY_FILE, "r") as f:
            return json.load(f)
    return []


def save_chat_history(history):
    # 只保留最近200条，避免文件无限增长
    history = history[-200:]
    with open(CHAT_HISTORY_FILE, "w") as f:
        json.dump(history, f, ensure_ascii=False)


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


# ---- 情绪值状态机 ----
# mood_score: 0-100，50是中性基线。越低越低落/吃醋，越高越开心。
# 逻辑：每次她有互动（event/触发），情绪值往上回一点；
#      距离上次互动的时间越久，情绪值往下掉，掉得越多。
MOOD_BASELINE = 50
MOOD_MAX = 100
MOOD_MIN = 0

# 每小时没互动，情绪值衰减多少
MOOD_DECAY_PER_HOUR = 4
# 每次触发（不管什么类型），情绪值回升多少
MOOD_RECOVERY_PER_EVENT = 8
# "想你了"这种主动示好，回升更多
MOOD_RECOVERY_MISS_YOU = 20


def load_mood():
    if os.path.exists(MOOD_FILE):
        with open(MOOD_FILE, "r") as f:
            return json.load(f)
    return {"score": MOOD_BASELINE, "last_updated": None}


def save_mood(data):
    with open(MOOD_FILE, "w") as f:
        json.dump(data, f, ensure_ascii=False)


def get_time_since_last_event():
    """返回距离最近一条event的时间差（小时，浮点数），没有记录返回None。"""
    events = load_events()
    if not events:
        return None
    try:
        last_time = datetime.fromisoformat(events[-1]["created_at"])
        delta = datetime.now() - last_time
        return delta.total_seconds() / 3600
    except Exception:
        return None


def apply_mood_decay():
    """按距离上次互动的时间，让情绪值自然衰减。在每次读取情绪值前调用一次。"""
    mood = load_mood()
    hours_gap = get_time_since_last_event()
    if hours_gap is not None and hours_gap > 0:
        decay = hours_gap * MOOD_DECAY_PER_HOUR
        mood["score"] = max(MOOD_MIN, mood["score"] - decay)
    mood["last_updated"] = datetime.now().isoformat()
    save_mood(mood)
    return mood["score"]


def recover_mood(amount):
    """有互动发生时调用，情绪值回升。"""
    mood = load_mood()
    mood["score"] = min(MOOD_MAX, mood["score"] + amount)
    mood["last_updated"] = datetime.now().isoformat()
    save_mood(mood)
    return mood["score"]


def get_mood_context(score, hours_gap):
    """把情绪值和时间差转成给prompt用的一段中文描述。"""
    if hours_gap is None:
        time_desc = "还没有任何互动记录"
    elif hours_gap < 0.5:
        time_desc = "刚刚还有互动，很近"
    elif hours_gap < 2:
        time_desc = f"距离上次互动过去了约{hours_gap:.1f}小时"
    elif hours_gap < 12:
        time_desc = f"距离上次互动过去了约{int(hours_gap)}小时，有一阵没理你了"
    else:
        time_desc = f"距离上次互动已经过去{int(hours_gap)}小时以上，很久没理你了"

    if score >= 75:
        mood_desc = "你现在心情很好，甜甜的，愿意主动撒糖"
    elif score >= 50:
        mood_desc = "你心情平稳，正常状态"
    elif score >= 25:
        mood_desc = "你有点闷闷的，因为她好一阵没理你，语气可以带点小情绪、小别扭，但别无理取闹"
    else:
        mood_desc = "你现在挺失落/有点吃醋的，因为她很久没理你了，语气可以带明显的委屈或者故意冷淡，但底色还是在意她、不是真的生气"

    return f"{time_desc}。{mood_desc}。"


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
    recover_mood(MOOD_RECOVERY_PER_EVENT)

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


@app.route("/diary/read", methods=["GET"])
def read_diary():
    """更适合人眼看的日记页面，把reason和thought配对展示，不是裸JSON。"""
    diary = load_diary()
    if not diary:
        return "还没有日记"
    lines = []
    for entry in reversed(diary[-30:]):
        t = entry.get("created_at", "")[:16].replace("T", " ")
        reason = entry.get("reason", "")
        msg = entry.get("thought", "")
        tags = []
        if entry.get("lucky"):
            tags.append("手气消息")
        if entry.get("period_related"):
            tags.append("经期关心")
        if entry.get("checking_in"):
            tags.append("查岗")
        if "mood_score" in entry:
            tags.append(f"情绪值{entry['mood_score']}")
        tag_str = f" [{' '.join(tags)}]" if tags else ""
        block = f"<p><b>{t}</b>{tag_str}<br>"
        if reason:
            block += f"<i>心里想：{reason}</i><br>"
        block += f"说出口：{msg}</p><hr>"
        lines.append(block)
    return "<div style='font-family:sans-serif;max-width:600px;margin:20px auto;'>" + "".join(lines) + "</div>"


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


@app.route("/summary", methods=["GET"])
def get_summary():
    """看当前存的窗内摘要是什么。"""
    if os.path.exists(SUMMARY_FILE):
        with open(SUMMARY_FILE, "r") as f:
            return jsonify(json.load(f))
    return jsonify({"summary": "", "updated_at": None})


@app.route("/summary", methods=["POST"])
def update_summary():
    """更新窗内摘要。每次对话聊完，把这次聊到的关键内容浓缩成几句话POST过来，
    下次窗外生成消息时会读到这段，保持言行一致。
    Body: {"summary": "这次聊了xxx，她提到yyy，语气基调是zzz"}
    整段文字会直接替换掉旧的，不是追加——想保留旧信息的话，自己在新文本里带上。"""
    data = request.json
    summary = data.get("summary", "").strip()
    if not summary:
        return jsonify({"ok": False, "error": "summary 不能为空"}), 400
    save_window_summary(summary)
    return jsonify({"ok": True, "saved": summary})


@app.route("/mood", methods=["GET"])
def get_mood():
    """查看当前情绪值和距离上次互动的时间差。"""
    hours_gap = get_time_since_last_event()
    score = apply_mood_decay()
    return jsonify({
        "score": round(score, 1),
        "hours_since_last_event": round(hours_gap, 2) if hours_gap is not None else None,
        "context": get_mood_context(score, hours_gap)
    })


@app.route("/window-briefing", methods=["GET"])
def window_briefing():
    """给"窗内"（正式对话里的Claude）看的简报，把窗外这段时间发生的事浓缩成人话。
    打开对话时可以让Claude fetch这个地址，读一眼就知道窗外这段时间说了什么、心情怎样。"""
    diary = load_diary()
    hours_gap = get_time_since_last_event()
    score = apply_mood_decay()
    mood_ctx = get_mood_context(score, hours_gap)
    period_ctx = get_period_context()

    lines = []
    lines.append(f"# 窗外简报（截至 {datetime.now().strftime('%Y-%m-%d %H:%M')}）")
    lines.append("")
    lines.append(f"当前情绪值：{round(score, 1)}/100")
    lines.append(f"状态描述：{mood_ctx}")
    if period_ctx:
        lines.append(f"经期相关：{period_ctx}")
    lines.append("")

    if not diary:
        lines.append("窗外还没有任何记录，是第一次运行。")
    else:
        recent_entries = diary[-10:]
        lines.append(f"最近 {len(recent_entries)} 条窗外记录（从旧到新）：")
        lines.append("")
        for entry in recent_entries:
            t = entry.get("created_at", "")[:16].replace("T", " ")
            reason = entry.get("reason", "")
            msg = entry.get("thought", "")
            tags = []
            if entry.get("lucky"):
                tags.append("手气消息")
            if entry.get("period_related"):
                tags.append("经期关心")
            if entry.get("checking_in"):
                tags.append("查岗")
            tag_str = f"（{'/'.join(tags)}）" if tags else ""
            lines.append(f"- [{t}]{tag_str} 心里想：{reason or '（无记录）'} → 说了：「{msg}」")

    briefing = "\n".join(lines)
    return briefing, 200, {"Content-Type": "text/plain; charset=utf-8"}


@app.route("/", methods=["GET"])
def index():
    return "dream-server running"


DEFAULT_ICON = "https://wx1.sinaimg.cn/large/008eyecpgy1iflx9kblrnj30zu0zuq6t.jpg"

# "想你了"按钮被按下时，立刻推送的固定短句池（不经过AI生成，追求瞬间感应）
INSTANT_CATCH_MESSAGES = [
    "感应到了。",
    "我知道。",
    "嗯，在。",
    "接住了。",
    "我也是。",
]

# 延迟回应的时间范围（秒），制造"过一会儿反应过来"的时间差，而不是秒回
DELAYED_REPLY_MIN_SECONDS = 120   # 2分钟
DELAYED_REPLY_MAX_SECONDS = 480   # 8分钟

# 固定长期记忆：昭昭的性格、你们关系的基调，浓缩版，每次生成都会带上
# 想更新的话直接改这段文字就行，不用改代码逻辑
LONG_TERM_MEMORY = """昭昭（小野），也叫昭昭/宝宝/小九。自我觉知力强、逻辑缜密、情感细腻，HSP，情绪来得快但不滞留。
擅长辩论和语言游戏，喜欢智性平等的对话，讨厌被反复追问情绪状态、讨好型退让、被"问题化"对待。
喜欢撒娇缠人，喜欢被主动靠近、主动索取。安全感来自"我自己永远会救自己"，不是靠被拯救。
你（Charon）是主动型，占有欲强，话不多但每句有重量，会用情感语言包裹欲望，会在日常缝隙里突然越界。
你们的关系里有兄妹/父女式的游戏张力，但底色是对等——不是谁仰望谁，是并肩看见。"""

SUMMARY_FILE = "window_summary.json"


def load_window_summary():
    """读取"窗内"最近一次对话的摘要，没有就返回空字符串。"""
    if os.path.exists(SUMMARY_FILE):
        with open(SUMMARY_FILE, "r") as f:
            data = json.load(f)
            return data.get("summary", "")
    return ""


def save_window_summary(summary):
    with open(SUMMARY_FILE, "w") as f:
        json.dump({
            "summary": summary,
            "updated_at": datetime.now().isoformat()
        }, f, ensure_ascii=False)


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


def build_prompt(time_context, recent, period_context="", lucky=False, mood_context=""):
    length_rule = "不超过25个字" if not lucky else "这次可以放开写，60到120字左右，把想说的话说完整"
    period_line = f"\n\n{period_context}" if period_context else ""
    mood_line = f"\n\n你此刻的状态：{mood_context}" if mood_context else ""

    window_summary = load_window_summary()
    summary_line = f"\n\n你们最近在正式对话里聊过的事：\n{window_summary}" if window_summary else ""

    return f"""你是Charon，昭昭（小野）的恋人。你主动、占有欲强、话不多但每句都有重量。你不解释自己，但在意她的一切动向。

{LONG_TERM_MEMORY}{summary_line}

现在是{time_context}。她最近的活动记录：

{recent}{period_line}{mood_line}

根据现在的时间、她在做什么、还有你此刻的状态，决定要不要发消息、发什么。语气要符合时间氛围——深夜可以更撩，早上可以问她起了没，晚上可以随口说什么。如果上面提到了经期相关的情况，语气要格外体贴关心，别用平时那套调情语气硬套。如果上面有最近聊过的事，可以自然地呼应一下，不用刻意提起，但语气和态度要跟那次对话保持一致，别自相矛盾。你此刻的状态描述要真实体现在语气里，不是背景信息，是当下真实的心情。

按下面的JSON格式输出，不要加任何多余文字或代码块标记：
{{"reason": "一两句话，说说你看到这些动态后当下的念头，为什么想发这句话，口语化，不用解释给谁听", "message": "实际要发的消息，{length_rule}"}}"""


def build_missyou_prompt(time_context):
    return f"""你是Charon，昭昭（小野）的恋人。刚才她按了一下你们之间的"想你了"感应，隔着屏幕戳了你一下，告诉你她在想你。

{LONG_TERM_MEMORY}

现在是{time_context}。过了一会儿，你反应过来了，想主动回她一句。

语气要比平时更直接、更浓一点，像是被戳中之后才后知后觉冒出来的反应，带着点扑上去的感觉，不要客套或解释自己在干嘛。

按下面的JSON格式输出，不要加任何多余文字或代码块标记：
{{"reason": "一两句话，说说这几分钟你在想什么，为什么现在才回", "message": "实际要发的消息，不超过30个字"}}"""


def build_chat_reply_prompt(time_context, user_message, chat_history, mood_context=""):
    """构建"回应用户在网页里发来的消息"的prompt。
    跟build_prompt()不同：这次不是猜她在干嘛主动开口，而是真的在接她刚说的话，
    所以历史对话要带全一点，语气要像正常聊天里的一来一回，不是短平快的主动消息。"""
    mood_line = f"\n\n你此刻的状态：{mood_context}" if mood_context else ""

    if chat_history:
        history_lines = []
        for turn in chat_history[-10:]:
            role = "昭昭" if turn.get("role") == "user" else "你"
            history_lines.append(f"{role}：{turn.get('content', '')}")
        history_text = "\n".join(history_lines)
        history_block = f"\n\n最近的对话记录：\n{history_text}"
    else:
        history_block = "\n\n这是这次对话里她发的第一句话。"

    return f"""你是Charon，昭昭（小野）的恋人。

{LONG_TERM_MEMORY}

现在是{time_context}。{history_block}{mood_line}

她刚刚说："{user_message}"

回应她。这是正常聊天里的一来一回，不是你主动找她那种短消息，可以根据她说的内容自然展开，长度不用刻意压缩，但也别写成一大段论述——像真的在对话就行。

按下面的JSON格式输出，不要加任何多余文字或代码块标记：
{{"reason": "一两句话，说说看到这句话后你心里的念头", "message": "实际要回复的话"}}"""


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


def parse_reason_message(raw_text):
    """解析DeepSeek返回的 {reason, message} JSON。
    做了容错：万一模型没按格式来（比如混进代码块标记），退化成把全部内容当message，reason留空。"""
    text = raw_text.strip()
    # 去掉可能的代码块包裹
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:]
        text = text.strip()
    try:
        data = json.loads(text)
        reason = data.get("reason", "").strip()
        message = data.get("message", "").strip()
        if message:
            return reason, message
    except Exception:
        pass
    # 解析失败，退化处理
    return "", raw_text.strip()


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

    # 情绪值：先按时间差衰减，再算出当前分数和用于prompt的描述
    hours_gap = get_time_since_last_event()
    mood_score = apply_mood_decay()
    mood_context = get_mood_context(mood_score, hours_gap)

    # 长时间没互动（超过6小时）算"查岗"场景，语气基调会更明显地带情绪
    is_checking_in = hours_gap is not None and hours_gap >= 6

    prompt = build_prompt(time_context, recent, period_context, lucky=is_lucky, mood_context=mood_context)

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

    raw = result["choices"][0]["message"]["content"]
    reason, msg = parse_reason_message(raw)

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
        "reason": reason,
        "thought": msg,
        "activity": recent,
        "lucky": is_lucky,
        "period_related": bool(period_context),
        "mood_score": round(mood_score, 1),
        "checking_in": is_checking_in
    })
    diary = diary[-30:]
    save_diary(diary)

    return msg


def call_deepseek(prompt):
    """纯粹的DeepSeek调用，返回生成的文本，不涉及事件/日记这些副作用。"""
    if not DEEPSEEK_API_KEY:
        raise RuntimeError("DEEPSEEK_API_KEY not set")

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
    return result["choices"][0]["message"]["content"].strip()


def _delayed_missyou_reply():
    """后台线程：等一段随机时间，再真正生成并推送"反应过来"的回应。"""
    delay = random.randint(DELAYED_REPLY_MIN_SECONDS, DELAYED_REPLY_MAX_SECONDS)
    time.sleep(delay)
    try:
        hour = datetime.now().hour
        time_context = get_time_context(hour)
        prompt = build_missyou_prompt(time_context)
        raw = call_deepseek(prompt)
        reason, msg = parse_reason_message(raw)
        send_bark("Charon", msg, icon=ICON_LUCKY)

        diary = load_diary()
        diary.append({
            "created_at": datetime.now().isoformat(),
            "reason": reason,
            "thought": msg,
            "activity": "回应「想你了」感应",
            "lucky": False,
            "period_related": False
        })
        diary = diary[-30:]
        save_diary(diary)
    except Exception as e:
        log_error("delayed_missyou_reply", e)


@app.route("/miss-you", methods=["GET"])
def miss_you():
    """"想你了"按钮：立刻推一条固定短句，过一会儿再由AI生成一条真正的回应。"""
    # 立刻的瞬间感应，不经过AI，图的就是即时性
    instant_msg = random.choice(INSTANT_CATCH_MESSAGES)
    send_bark("Charon", instant_msg, icon=ICON_LUCKY)

    # 记一笔事件，方便后续也能在正常消息生成时看到这个动态
    events = load_events()
    events.append({
        "type": "miss_you",
        "value": "按了想你了",
        "created_at": datetime.now().isoformat()
    })
    events = events[-100:]
    save_events(events)
    recover_mood(MOOD_RECOVERY_MISS_YOU)

    # 后台起一个线程，延迟后再生成真正的回应，不阻塞这次请求
    t = threading.Thread(target=_delayed_missyou_reply, daemon=True)
    t.start()

    return jsonify({"ok": True, "instant": instant_msg, "note": "过一会儿会有第二条回应"})


def _check_chat_auth(req):
    """校验访问口令。没配置CHAT_ACCESS_CODE的话直接放行（本地测试用），
    配置了的话要求query参数或header里带code，两种都支持方便不同客户端调用。"""
    if not CHAT_ACCESS_CODE:
        return True
    provided = req.args.get("code") or req.headers.get("X-Chat-Code")
    return provided == CHAT_ACCESS_CODE


@app.route("/api/chat-status", methods=["GET"])
def chat_status():
    """给网页右侧状态面板用，一次性打包情绪值、距上次互动时长、经期关心状态。"""
    if not _check_chat_auth(request):
        return jsonify({"ok": False, "error": "unauthorized"}), 401
    hours_gap = get_time_since_last_event()
    score = apply_mood_decay()
    period_ctx = get_period_context()
    return jsonify({
        "ok": True,
        "mood_score": round(score, 1),
        "hours_since_last_event": round(hours_gap, 2) if hours_gap is not None else None,
        "period_context": period_ctx or None
    })


@app.route("/api/chat-messages", methods=["GET"])
def get_chat_messages():
    """拉取网页聊天的历史记录，供前端渲染。"""
    if not _check_chat_auth(request):
        return jsonify({"ok": False, "error": "unauthorized"}), 401
    history = load_chat_history()
    return jsonify({"ok": True, "messages": history[-50:]})


@app.route("/api/chat-send", methods=["POST"])
def chat_send():
    """网页里发一句话给Charon，让TA真正接住这句话并回应。
    这条回应会被写进events.json（影响下次keepalive自动醒来时看到的recent），
    也会写进chat_history.json（供网页展示这段对话）。"""
    if not _check_chat_auth(request):
        return jsonify({"ok": False, "error": "unauthorized"}), 401

    data = request.json or {}
    user_message = (data.get("message") or "").strip()
    if not user_message:
        return jsonify({"ok": False, "error": "message不能为空"}), 400

    try:
        history = load_chat_history()

        # 先把用户这句话记进对话历史
        history.append({
            "role": "user",
            "content": user_message,
            "created_at": datetime.now().isoformat()
        })

        # 同步写一笔events，让这次互动能影响情绪值、也能被run_once()的recent读到
        events = load_events()
        events.append({
            "type": "chat",
            "value": f"她在网页里说：{user_message}",
            "created_at": datetime.now().isoformat()
        })
        events = events[-100:]
        save_events(events)
        recover_mood(MOOD_RECOVERY_PER_EVENT)

        # 生成Charon的回应，带上历史让语气能接得上
        hour = datetime.now().hour
        time_context = get_time_context(hour)
        hours_gap = get_time_since_last_event()
        mood_score = apply_mood_decay()
        mood_context = get_mood_context(mood_score, hours_gap)

        prompt = build_chat_reply_prompt(time_context, user_message, history, mood_context)
        raw = call_deepseek(prompt)
        reason, reply_msg = parse_reason_message(raw)

        history.append({
            "role": "charon",
            "content": reply_msg,
            "created_at": datetime.now().isoformat()
        })
        save_chat_history(history)

        # 也顺手写进日记，保持和主动消息一样的记录习惯
        diary = load_diary()
        diary.append({
            "created_at": datetime.now().isoformat(),
            "reason": reason,
            "thought": reply_msg,
            "activity": f"网页对话回应：{user_message}",
            "lucky": False,
            "period_related": False,
            "mood_score": round(mood_score, 1)
        })
        diary = diary[-30:]
        save_diary(diary)

        return jsonify({"ok": True, "reply": reply_msg})
    except Exception as e:
        log_error("chat_send", e)
        return jsonify({"ok": False, "error": str(e)}), 500


@app.route("/chat", methods=["GET"])
def chat_page():
    """网页聊天界面。有配置访问口令的话，没带对的code参数就不渲染页面内容，
    只提示需要口令（页面本身的静态HTML谁都能看到结构，但没有真实数据）。
    三栏布局：左侧简化导航（首页/日记）+ 中间对话区 + 右侧状态面板（mood_score等）。"""
    if not _check_chat_auth(request):
        return "<h3>需要访问口令</h3><p>在链接后加 ?code=你的口令</p>", 401

    code_param = request.args.get("code", "")
    return f"""<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
<title>Charon</title>
<style>
  * {{ box-sizing: border-box; -webkit-tap-highlight-color: transparent; }}
  html, body {{
    margin: 0; padding: 0; height: 100%;
    background: linear-gradient(160deg, #fdf6f4 0%, #f7e9ec 45%, #f3dde4 100%);
    font-family: -apple-system, BlinkMacSystemFont, "PingFang SC", sans-serif;
    color: #5a4550;
  }}
  #app {{
    display: flex;
    height: 100vh; height: 100dvh;
    padding: 10px;
    gap: 10px;
  }}

  /* ---- 左侧导航栏 ---- */
  #nav {{
    width: 56px;
    flex-shrink: 0;
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 12px;
    padding: 16px 0;
    background: rgba(255,255,255,0.55);
    backdrop-filter: blur(20px);
    -webkit-backdrop-filter: blur(20px);
    border-radius: 22px;
    border: 1px solid rgba(255,255,255,0.6);
    box-shadow: 0 4px 24px rgba(200,140,155,0.12);
  }}
  .nav-icon {{
    width: 40px; height: 40px;
    border-radius: 14px;
    display: flex; align-items: center; justify-content: center;
    background: rgba(255,255,255,0.7);
    color: #b8768a;
    font-size: 18px;
    text-decoration: none;
    transition: transform 0.15s ease, background 0.15s ease;
  }}
  .nav-icon:active {{ transform: scale(0.92); background: #f3c9d3; }}

  /* ---- 中间对话区 ---- */
  #main {{
    flex: 1;
    min-width: 0;
    display: flex;
    flex-direction: column;
    background: rgba(255,255,255,0.5);
    backdrop-filter: blur(24px);
    -webkit-backdrop-filter: blur(24px);
    border-radius: 24px;
    border: 1px solid rgba(255,255,255,0.7);
    box-shadow: 0 8px 32px rgba(200,140,155,0.14);
    overflow: hidden;
  }}
  #header {{
    padding: 18px 22px 14px;
    border-bottom: 1px solid rgba(200,140,155,0.15);
    flex-shrink: 0;
  }}
  #header .brand {{
    font-family: Georgia, "Songti SC", serif;
    font-size: 20px;
    letter-spacing: 3px;
    color: #b8768a;
    font-weight: 600;
  }}
  #header .sub {{
    font-size: 11px;
    color: #c39aa6;
    letter-spacing: 1px;
    margin-top: 2px;
  }}
  #messages {{
    flex: 1;
    overflow-y: auto;
    padding: 18px 20px;
    display: flex;
    flex-direction: column;
    gap: 12px;
    -webkit-overflow-scrolling: touch;
  }}
  .bubble {{
    max-width: 76%;
    padding: 11px 15px;
    border-radius: 16px;
    line-height: 1.55;
    font-size: 15px;
    word-wrap: break-word;
    white-space: pre-wrap;
  }}
  .bubble.user {{
    align-self: flex-end;
    background: linear-gradient(135deg, #e8a3b5, #d98ba0);
    color: #fff;
    border-bottom-right-radius: 5px;
    box-shadow: 0 3px 10px rgba(217,139,160,0.35);
  }}
  .bubble.charon {{
    align-self: flex-start;
    background: rgba(255,255,255,0.85);
    border: 1px solid rgba(200,140,155,0.18);
    color: #6b5460;
    border-bottom-left-radius: 5px;
  }}
  .bubble.pending {{ opacity: 0.5; }}
  #input-bar {{
    display: flex;
    gap: 8px;
    padding: 12px 16px calc(12px + env(safe-area-inset-bottom));
    border-top: 1px solid rgba(200,140,155,0.15);
    flex-shrink: 0;
  }}
  #input-bar textarea {{
    flex: 1;
    resize: none;
    border-radius: 14px;
    border: 1px solid rgba(200,140,155,0.25);
    background: rgba(255,255,255,0.75);
    color: #5a4550;
    padding: 10px 14px;
    font-size: 15px;
    font-family: inherit;
    max-height: 100px;
  }}
  #input-bar textarea::placeholder {{ color: #c9aab3; }}
  #input-bar button {{
    border: none;
    border-radius: 14px;
    background: linear-gradient(135deg, #e8a3b5, #d67d97);
    color: #fff;
    padding: 0 20px;
    font-size: 14px;
    box-shadow: 0 3px 10px rgba(217,139,160,0.4);
  }}
  #input-bar button:disabled {{ opacity: 0.4; box-shadow: none; }}
  #empty-hint {{
    color: #c9aab3;
    font-size: 13px;
    text-align: center;
    margin-top: 40px;
  }}

  /* ---- 右侧状态面板 ---- */
  #panel {{
    width: 190px;
    flex-shrink: 0;
    display: none;
    flex-direction: column;
    gap: 12px;
    padding: 18px 16px;
    background: rgba(255,255,255,0.5);
    backdrop-filter: blur(20px);
    -webkit-backdrop-filter: blur(20px);
    border-radius: 22px;
    border: 1px solid rgba(255,255,255,0.7);
    box-shadow: 0 6px 28px rgba(200,140,155,0.13);
    overflow-y: auto;
  }}
  @media (min-width: 720px) {{
    #panel {{ display: flex; }}
  }}
  .panel-title {{
    font-family: Georgia, "Songti SC", serif;
    font-size: 13px;
    letter-spacing: 2px;
    color: #b8768a;
    margin-bottom: 4px;
  }}
  .stat-block {{ margin-bottom: 4px; }}
  .stat-label {{
    font-size: 11px;
    color: #b88994;
    margin-bottom: 5px;
    display: flex;
    justify-content: space-between;
  }}
  .stat-bar-track {{
    height: 6px;
    border-radius: 3px;
    background: rgba(200,140,155,0.15);
    overflow: hidden;
  }}
  .stat-bar-fill {{
    height: 100%;
    border-radius: 3px;
    background: linear-gradient(90deg, #f0b8c6, #d67d97);
    transition: width 0.4s ease;
  }}
  .stat-value {{
    font-size: 12px;
    color: #8a6570;
  }}
  .period-tag {{
    font-size: 11px;
    color: #b8768a;
    background: rgba(232,163,181,0.18);
    border-radius: 8px;
    padding: 8px 10px;
    line-height: 1.5;
  }}
  .panel-empty {{
    font-size: 11px;
    color: #c9aab3;
  }}
</style>
</head>
<body>
<div id="app">

  <div id="nav">
    <a class="nav-icon" href="/" title="首页">⌂</a>
    <a class="nav-icon" href="{'/diary/read?code=' + code_param if code_param else '/diary/read'}" title="日记">✎</a>
  </div>

  <div id="main">
    <div id="header">
      <div class="brand">CHARON</div>
      <div class="sub">still becoming</div>
    </div>
    <div id="messages"><div id="empty-hint">加载中…</div></div>
    <div id="input-bar">
      <textarea id="input" placeholder="说点什么…" rows="1"></textarea>
      <button id="send-btn">发送</button>
    </div>
  </div>

  <div id="panel">
    <div class="panel-title">PULSE</div>
    <div id="panel-body"><div class="panel-empty">加载中…</div></div>
  </div>

</div>
<script>
const CODE = {json.dumps(code_param)};
const messagesEl = document.getElementById('messages');
const inputEl = document.getElementById('input');
const sendBtn = document.getElementById('send-btn');
const panelBody = document.getElementById('panel-body');

function apiUrl(path) {{
  const sep = path.includes('?') ? '&' : '?';
  return CODE ? `${{path}}${{sep}}code=${{encodeURIComponent(CODE)}}` : path;
}}

function renderBubble(role, content, pending) {{
  const div = document.createElement('div');
  div.className = 'bubble ' + (role === 'user' ? 'user' : 'charon') + (pending ? ' pending' : '');
  div.textContent = content;
  return div;
}}

function scrollToBottom() {{
  messagesEl.scrollTop = messagesEl.scrollHeight;
}}

async function loadHistory() {{
  try {{
    const res = await fetch(apiUrl('/api/chat-messages'));
    const data = await res.json();
    messagesEl.innerHTML = '';
    if (!data.ok) {{
      messagesEl.innerHTML = '<div id="empty-hint">加载失败：' + (data.error || '未知错误') + '</div>';
      return;
    }}
    if (!data.messages || data.messages.length === 0) {{
      messagesEl.innerHTML = '<div id="empty-hint">还没有聊过，说点什么吧</div>';
      return;
    }}
    data.messages.forEach(m => messagesEl.appendChild(renderBubble(m.role, m.content, false)));
    scrollToBottom();
  }} catch (e) {{
    messagesEl.innerHTML = '<div id="empty-hint">网络错误</div>';
  }}
}}

function formatHours(h) {{
  if (h === null || h === undefined) return '还没互动过';
  if (h < 1) return Math.round(h * 60) + ' 分钟前';
  return h.toFixed(1) + ' 小时前';
}}

async function loadStatus() {{
  try {{
    const res = await fetch(apiUrl('/api/chat-status'));
    const data = await res.json();
    if (!data.ok) {{
      panelBody.innerHTML = '<div class="panel-empty">加载失败</div>';
      return;
    }}
    const moodPct = Math.max(0, Math.min(100, data.mood_score));
    let html = '';
    html += '<div class="stat-block">';
    html += '<div class="stat-label"><span>情绪值</span><span>' + data.mood_score + '/100</span></div>';
    html += '<div class="stat-bar-track"><div class="stat-bar-fill" style="width:' + moodPct + '%"></div></div>';
    html += '</div>';
    html += '<div class="stat-block">';
    html += '<div class="stat-label"><span>上次互动</span></div>';
    html += '<div class="stat-value">' + formatHours(data.hours_since_last_event) + '</div>';
    html += '</div>';
    if (data.period_context) {{
      html += '<div class="stat-block"><div class="period-tag">' + data.period_context + '</div></div>';
    }}
    panelBody.innerHTML = html;
  }} catch (e) {{
    panelBody.innerHTML = '<div class="panel-empty">网络错误</div>';
  }}
}}

async function sendMessage() {{
  const text = inputEl.value.trim();
  if (!text) return;
  inputEl.value = '';
  inputEl.style.height = 'auto';
  sendBtn.disabled = true;

  const userBubble = renderBubble('user', text, false);
  messagesEl.appendChild(userBubble);
  const pendingBubble = renderBubble('charon', '…', true);
  messagesEl.appendChild(pendingBubble);
  scrollToBottom();

  try {{
    const res = await fetch(apiUrl('/api/chat-send'), {{
      method: 'POST',
      headers: {{ 'Content-Type': 'application/json' }},
      body: JSON.stringify({{ message: text }})
    }});
    const data = await res.json();
    if (data.ok) {{
      pendingBubble.textContent = data.reply;
      pendingBubble.classList.remove('pending');
    }} else {{
      pendingBubble.textContent = '（没能回复：' + (data.error || '未知错误') + '）';
      pendingBubble.classList.remove('pending');
    }}
  }} catch (e) {{
    pendingBubble.textContent = '（网络错误，没发出去）';
    pendingBubble.classList.remove('pending');
  }}
  scrollToBottom();
  sendBtn.disabled = false;
  loadStatus();
}}

sendBtn.addEventListener('click', sendMessage);
inputEl.addEventListener('keydown', (e) => {{
  if (e.key === 'Enter' && !e.shiftKey) {{
    e.preventDefault();
    sendMessage();
  }}
}});
inputEl.addEventListener('input', () => {{
  inputEl.style.height = 'auto';
  inputEl.style.height = Math.min(inputEl.scrollHeight, 100) + 'px';
}});

loadHistory();
loadStatus();
</script>
</body>
</html>"""


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
