import subprocess
subprocess.run(["pip", "install", "requests"], capture_output=True)

from flask import Flask, request, jsonify
from datetime import datetime, date
import json, os, requests, threading, time, traceback, random

app = Flask(__name__)

# 所有数据文件统一放在 DATA_DIR 指向的目录下。
# 配了 Railway Volume 的话，把 DATA_DIR 设成 Volume 的挂载路径（比如 /data），
# 这样重新部署容器时这些文件不会跟着容器一起被清空。
# 没配置的话默认用当前目录（本地测试用，重新部署照样会丢，这是预期行为）。
DATA_DIR = os.environ.get("DATA_DIR", ".")
os.makedirs(DATA_DIR, exist_ok=True)


def data_path(filename):
    return os.path.join(DATA_DIR, filename)


EVENTS_FILE = data_path("events.json")
DIARY_FILE = data_path("diary.json")
ERROR_LOG = data_path("error.log")
PERIOD_FILE = data_path("period.json")
CHAT_HISTORY_FILE = data_path("chat_history.json")
MOOD_FILE = data_path("mood.json")
SUMMARY_FILE = data_path("window_summary.json")

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


def new_msg_id():
    """给每条聊天消息生成一个唯一ID，用于前端指定删除某一条。
    用时间戳+随机数拼接，不需要额外依赖（不用uuid库也够用，量级不大）。"""
    return f"{int(time.time() * 1000)}-{random.randint(1000, 9999)}"


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

# 网页聊天用的头像（灰兔=Charon，粉兔=昭昭）。放在 static/ 目录下随代码一起部署，
# 不依赖任何第三方图床（新浪图床等对外链有防盗链限制，会导致图片显示不出来）。
CHAT_AVATAR_CHARON = "/static/avatar_charon.jpg"
CHAT_AVATAR_USER = "/static/avatar_user.jpg"

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
        for turn in chat_history[-20:]:
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
    """给网页右侧状态面板和header状态文字用，一次性打包所有能展示的状态数据。
    这些数据后端本来就有（情绪值/经期/日记/摘要），这里只是集中暴露出来给前端展示。"""
    if not _check_chat_auth(request):
        return jsonify({"ok": False, "error": "unauthorized"}), 401

    hours_gap = get_time_since_last_event()
    score = apply_mood_decay()
    period_ctx = get_period_context()

    # 查岗状态：距上次互动超过6小时，跟run_once()里的判断口径保持一致
    is_checking_in = hours_gap is not None and hours_gap >= 6

    # 最近一条日记：拿reason（TA当时心里想的）和lucky标记
    diary = load_diary()
    last_entry = diary[-1] if diary else None
    last_thought = last_entry.get("reason") if last_entry else None
    last_was_lucky = bool(last_entry.get("lucky")) if last_entry else False

    # 窗内摘要：正式对话里聊过的内容
    window_summary = load_window_summary()

    # 今日互动次数：数events.json里created_at是今天的条数
    events = load_events()
    today_str = date.today().isoformat()
    today_count = sum(1 for e in events if e.get("created_at", "").startswith(today_str))

    return jsonify({
        "ok": True,
        "mood_score": round(score, 1),
        "status_label": get_chat_status_label(score),
        "hours_since_last_event": round(hours_gap, 2) if hours_gap is not None else None,
        "period_context": period_ctx or None,
        "is_checking_in": is_checking_in,
        "last_thought": last_thought or None,
        "last_was_lucky": last_was_lucky,
        "window_summary": window_summary or None,
        "today_interaction_count": today_count
    })


@app.route("/api/chat-messages", methods=["GET"])
def get_chat_messages():
    """拉取网页聊天的历史记录，供前端渲染。"""
    if not _check_chat_auth(request):
        return jsonify({"ok": False, "error": "unauthorized"}), 401
    history = load_chat_history()
    return jsonify({"ok": True, "messages": history[-50:]})


@app.route("/api/chat-delete", methods=["POST"])
def chat_delete():
    """删除网页聊天里的某一条消息（按id匹配）。
    只删chat_history.json里的这一条；如果这条是"user"发的话，
    顺手尝试从events.json里删掉内容和时间都对得上的那条同步记录，
    避免Charon下次醒来时recent里还看得到已经删掉的话。
    注意：events.json里没有存消息id，只能按"value包含这句话内容+created_at相同"来匹配，
    不是绝对精确（极小概率误删同一秒内说的相同内容），但日常使用够用。"""
    if not _check_chat_auth(request):
        return jsonify({"ok": False, "error": "unauthorized"}), 401

    data = request.json or {}
    msg_id = data.get("id")
    if not msg_id:
        return jsonify({"ok": False, "error": "缺少id参数"}), 400

    try:
        history = load_chat_history()
        target = next((m for m in history if m.get("id") == msg_id), None)
        if not target:
            return jsonify({"ok": False, "error": "没找到这条消息，可能已经被删过了"}), 404

        history = [m for m in history if m.get("id") != msg_id]
        save_chat_history(history)

        # 同步清理events.json里对应的那条（仅针对用户发的消息，Charon的回复不会写进events）
        if target.get("role") == "user":
            events = load_events()
            target_created_at = target.get("created_at", "")
            target_content = target.get("content", "")
            events = [
                e for e in events
                if not (
                    e.get("type") == "chat"
                    and e.get("created_at") == target_created_at
                    and target_content in e.get("value", "")
                )
            ]
            save_events(events)

        return jsonify({"ok": True, "deleted_id": msg_id})
    except Exception as e:
        log_error("chat_delete", e)
        return jsonify({"ok": False, "error": str(e)}), 500


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
        user_msg_id = new_msg_id()
        history.append({
            "id": user_msg_id,
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

        charon_msg_id = new_msg_id()
        history.append({
            "id": charon_msg_id,
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

        return jsonify({
            "ok": True,
            "reply": reply_msg,
            "user_msg_id": user_msg_id,
            "charon_msg_id": charon_msg_id
        })
    except Exception as e:
        log_error("chat_send", e)
        return jsonify({"ok": False, "error": str(e)}), 500


def get_chat_status_label(score):
    """网页聊天header里显示的状态短语，跟get_mood_context()的详细描述不同，
    这个要短、像个人在线状态那种感觉，一两个词就行。"""
    if score >= 75:
        return "心情不错"
    elif score >= 50:
        return "在线"
    elif score >= 25:
        return "有点安静"
    else:
        return "有点失落"


@app.route("/chat", methods=["GET"])
def chat_page():
    """网页聊天界面。有配置访问口令的话，没带对的code参数就不渲染页面内容，
    只提示需要口令（页面本身的静态HTML谁都能看到结构，但没有真实数据）。
    三栏布局：左侧简化导航（首页/日记）+ 中间对话区 + 右侧状态面板（mood_score等）。"""
    if not _check_chat_auth(request):
        return "<h3>需要访问口令</h3><p>在链接后加 ?code=你的口令</p>", 401

    code_param = request.args.get("code", "")
    # 将日记 URL 提前在 Python 阶段组装好，避免 f-string 内部嵌套复杂的三元运算产生解析 Bug
    diary_url = f"/diary/read?code={code_param}" if code_param else "/diary/read"

    return f"""<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
<title>Charon</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Dancing+Script:wght@600;700&display=swap" rel="stylesheet">
<style>
  * {{ box-sizing: border-box; -webkit-tap-highlight-color: transparent; }}
  
  html, body {{
    margin: 0; padding: 0; height: 100%;
    /* 最底层为柔和的灰粉色，衬托波点纹理 */
    background-color: #f7e6eb;
    font-family: "Songti SC", "STSong", Georgia, -apple-system, BlinkMacSystemFont, "PingFang SC", sans-serif;
    color: #5a4550;
    overflow: hidden;
  }}
  
  #app {{
    display: flex;
    height: 100vh; height: 100dvh;
    padding: 14px;
    gap: 14px;
    position: relative;
    z-index: 2;
  }}

  /* ---- 1. 左侧玻璃舱导航栏 (纯CSS纵向渐变波点网) ---- */
  #nav {{
    width: 64px;
    flex-shrink: 0;
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 16px;
    padding: 24px 0;
    border-radius: 20px;
    position: relative;
    background-color: #fff; /* 底色为白 */
    box-shadow: 0 8px 24px rgba(0, 0, 0, 0.05);
  }}
  /* 利用 mask-image 和渐变色叠加，生成 黑-粉-白-粉-黑 的波点列 */
  #nav::before {{
    content: ''; position: absolute; inset: 0; border-radius: 20px;
    background: linear-gradient(to bottom, #2a2a2a 0%, #d698a9 25%, #ffffff 50%, #d698a9 75%, #2a2a2a 100%);
    -webkit-mask-image: radial-gradient(circle, black 35%, transparent 40%);
    -webkit-mask-size: 14px 14px;
    mask-image: radial-gradient(circle, black 35%, transparent 40%);
    mask-size: 14px 14px;
    z-index: 0;
  }}
  .nav-icon {{
    position: relative; z-index: 1; /* 浮于波点之上 */
    width: 42px; height: 42px;
    border-radius: 50%; /* 圆形图标，契合波点美学 */
    display: flex; align-items: center; justify-content: center;
    background: #ffffff;
    color: #2a2a2a;
    font-size: 19px;
    text-decoration: none;
    border: 2px solid #2a2a2a; /* 硬核黑色细边框 */
    transition: all 0.25s ease;
  }}
  .nav-icon:hover {{
    background: #2a2a2a;
    color: #ffffff;
    transform: translateY(-2px);
  }}

  /* ---- 2. 中间对话区 ---- */
  #main {{
    flex: 1;
    min-width: 0;
    display: flex;
    flex-direction: column;
    border-radius: 24px;
    overflow: hidden;
    position: relative;
    background: #ffffff; /* 干净纯白底 */
    box-shadow: 0 10px 30px rgba(184, 118, 138, 0.15);
  }}
  
  /* 底部向上渐变的输入区波点纹理 */
  #main::after {{
    content: ''; position: absolute; bottom: 0; left: 0; right: 0; height: 160px;
    background: linear-gradient(to top, #2a2a2a 0%, #e6a8b8 40%, transparent 100%);
    -webkit-mask-image: radial-gradient(circle, black 35%, transparent 40%);
    -webkit-mask-size: 16px 16px;
    mask-image: radial-gradient(circle, black 35%, transparent 40%);
    mask-size: 16px 16px;
    z-index: 0; /* 在消息流底层 */
    pointer-events: none;
  }}

  /* 酷感黑色渐变 Header */
  #header {{
    padding: 20px 24px 18px;
    position: relative;
    display: flex;
    align-items: center;
    gap: 14px;
    z-index: 2;
    background: linear-gradient(90deg, #111111 0%, #2a2a2a 35%, rgba(255,255,255,0) 80%);
  }}
  /* Header 左侧黑底内的微弱浅色波点 */
  #header::before {{
    content: ''; position: absolute; top: 0; left: 0; width: 45%; height: 100%;
    background-image: radial-gradient(circle, rgba(255,255,255,0.12) 30%, transparent 35%);
    background-size: 10px 10px;
    -webkit-mask-image: linear-gradient(90deg, black 20%, transparent 100%);
    mask-image: linear-gradient(90deg, black 20%, transparent 100%);
    z-index: -1;
  }}
  
  #header-avatar {{
    width: 44px; height: 44px;
    border-radius: 50%;
    object-fit: cover;
    flex-shrink: 0;
    border: 1px solid rgba(255,255,255,0.3); /* 弱化头像白边 */
  }}
  
  #header-text {{ min-width: 0; flex: 1; }}
  
  /* 做旧金属立体拉丝质感字体 */
  #header .brand {{
    font-family: "Georgia", "Songti SC", serif;
    font-size: 28px;
    letter-spacing: 4px;
    font-weight: bold;
    background: linear-gradient(to bottom, #ffffff 0%, #cfc0c5 40%, #523b45 50%, #9e8a93 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    filter: drop-shadow(1px 1px 0px rgba(255,255,255,0.3)) drop-shadow(-1px -1px 0px rgba(0,0,0,0.8));
  }}
  
  /* 带有半透明磨砂白框的在线状态 */
  #header .sub {{
    display: inline-flex;
    align-items: center;
    gap: 5px;
    margin-top: 6px;
    background: rgba(255, 255, 255, 0.15); /* 黑色渐变上的微白框 */
    border: 1px solid rgba(255, 255, 255, 0.1);
    border-radius: 12px;
    padding: 3px 8px;
    font-size: 11px;
    color: #e2cbd4;
    letter-spacing: 1px;
  }}
  .status-dot {{
    width: 6px; height: 6px;
    border-radius: 50%;
    background: #84cc9a;
    box-shadow: 0 0 6px #84cc9a;
  }}
  .status-dot.low {{ 
    background: #c3a1ad; box-shadow: 0 0 6px #c3a1ad; 
  }}
  
  /* 右侧荧光花体签名 */
  .signature {{
    position: absolute;
    right: 24px; bottom: 16px;
    font-family: 'Dancing Script', cursive;
    color: #ffffff;
    font-size: 22px;
    opacity: 0.85;
    text-shadow: 0 2px 10px rgba(255, 255, 255, 0.5);
  }}

  /* ---- 消息流区 ---- */
  #messages {{
    flex: 1;
    overflow-y: auto;
    padding: 22px 24px;
    display: flex;
    flex-direction: column;
    gap: 18px;
    -webkit-overflow-scrolling: touch;
    position: relative;
    z-index: 1;
  }}
  .msg-row {{
    display: flex;
    align-items: flex-start;
    gap: 10px;
    max-width: 88%;
  }}
  .msg-row.user {{
    align-self: flex-end;
    flex-direction: row-reverse;
  }}
  .msg-row.charon {{
    align-self: flex-start;
  }}
  .msg-avatar {{
    width: 36px; height: 36px;
    border-radius: 50%;
    object-fit: cover;
    flex-shrink: 0;
    margin-top: 0;
  }}
  
  .msg-col {{
    display: flex;
    flex-direction: column;
    min-width: 0;
  }}
  /* 确保 msg-col 的子元素对齐方式正确，为时间戳贴边打下基础 */
  .msg-row.user .msg-col {{ align-items: flex-end; }}
  .msg-row.charon .msg-col {{ align-items: flex-start; }}
  
  .bubble {{
    padding: 12px 18px;
    line-height: 1.6;
    font-size: 15px;
    word-wrap: break-word;
    white-space: pre-wrap;
    /* 无边框、长按可触发撤回的极简气泡 */
    border: none; 
    cursor: default; /* 暗示可交互 */
  }}
  /* User 气泡：调浅了的高级暮粉渐变色 */
  .bubble.user {{
    background: linear-gradient(135deg, #e8a3b5, #d98ba0);
    color: #ffffff;
    border-radius: 18px 4px 18px 18px; 
    box-shadow: 0 4px 15px rgba(184, 96, 118, 0.15);
  }}
  .bubble.charon {{
    background: #f4eaed; /* 高级灰粉底色，取代单纯的半透明白 */
    color: #5a4550;
    border-radius: 4px 18px 18px 18px; 
  }}
  .bubble.pending {{ opacity: 0.5; }}
  
  /* 时间戳贴边修正：利用 align-self 绝对紧贴气泡左右边缘 */
  .msg-time {{
    font-size: 10px;
    color: #b08d98;
    margin-top: 4px;
  }}
  /* 左侧时间戳靠左 */
  .msg-row.charon .msg-time {{
    align-self: flex-start;
    margin-left: 2px;
  }}
  /* 右侧时间戳完美靠右 */
  .msg-row.user .msg-time {{
    align-self: flex-end;
    margin-right: 2px;
  }}

  /* ---- 输入栏 ---- */
  #input-bar {{
    display: flex;
    gap: 12px;
    padding: 16px 20px calc(16px + env(safe-area-inset-bottom));
    flex-shrink: 0;
    position: relative;
    z-index: 2; /* 盖在底部波点上 */
  }}
  /* 药丸式半透明白框输入栏 */
  #input-bar textarea {{
    flex: 1;
    resize: none;
    border-radius: 20px;
    border: none;
    background: rgba(255, 255, 255, 0.85); 
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.04);
    color: #5a4550;
    padding: 12px 18px;
    font-size: 15px;
    font-family: inherit;
    max-height: 100px;
    outline: none;
    transition: background 0.2s ease;
  }}
  #input-bar textarea:focus {{ background: #ffffff; }}
  #input-bar textarea::placeholder {{ color: #b08d98; }}
  
  /* 粉色发送药丸按钮 */
  #input-bar button {{
    border: none;
    border-radius: 20px;
    background: linear-gradient(135deg, #e8a3b5, #d98ba0); 
    color: #fff;
    padding: 0 24px;
    font-size: 14px;
    cursor: pointer;
    box-shadow: 0 4px 12px rgba(184, 96, 118, 0.2);
    transition: all 0.2s ease;
  }}
  #input-bar button:hover {{ transform: translateY(-1px); }}
  #input-bar button:active {{ transform: translateY(0); }}
  #input-bar button:disabled {{ opacity: 0.4; box-shadow: none; cursor: default; }}
  #empty-hint {{ color: #b08d98; font-size: 13px; text-align: center; margin-top: 40px; }}

  /* ---- 3. 右侧状态面板 (白底粉色波点) ---- */
  #panel {{
    width: 220px;
    flex-shrink: 0;
    display: none;
    flex-direction: column;
    gap: 16px;
    padding: 22px 20px;
    border-radius: 24px;
    overflow-y: auto;
    /* 白底，粉色圆点密铺 */
    background-color: #ffffff;
    background-image: radial-gradient(circle, #f9eaed 35%, transparent 36%);
    background-size: 12px 12px;
    box-shadow: 0 8px 24px rgba(0, 0, 0, 0.03);
  }}
  @media (min-width: 720px) {{ #panel {{ display: flex; }} }}
  
  .panel-title {{
    font-family: "Georgia", "Songti SC", serif;
    font-size: 15px;
    letter-spacing: 3px;
    color: #a66275;
    margin-bottom: 6px;
    text-transform: uppercase;
    font-weight: bold;
    background: rgba(255,255,255,0.7); /* 给文字垫一层微弱底色防止和波点糊在一起 */
    display: inline-block;
    padding-right: 4px;
  }}
  .stat-block {{ margin-bottom: 12px; }}
  .stat-label {{
    font-size: 11px;
    color: #b08d98;
    margin-bottom: 5px;
    display: flex;
    justify-content: space-between;
    align-items: baseline;
    background: rgba(255,255,255,0.7);
  }}
  
  /* 情绪值进度条：2px 极细直轨样式 */
  .stat-bar-track {{
    height: 2px; 
    border-radius: 0; 
    background: #efdbe1; /* 浅粉色细轨 */
    overflow: hidden;
    width: 100%;
  }}
  .stat-bar-fill {{
    height: 100%;
    border-radius: 0;
    background: #d98ba0; /* 深玫瑰粉进度条 */
    transition: width 0.4s ease;
  }}
  .stat-value {{ font-size: 11px; color: #b08d98; }}
  
  /* “心里话”板块：恢复原版优雅斜体、经典深玫瑰灰 */
  .thought-card {{
    font-size: 12.5px;
    color: #7a5a65; 
    background: rgba(255, 255, 255, 0.85); /* 较高的白底覆盖住背后波点，保证阅读性 */
    border: none; 
    border-radius: 16px;
    padding: 14px 16px;
    line-height: 1.6;
    font-style: italic; 
    box-shadow: 0 4px 12px rgba(184, 118, 138, 0.08); 
  }}
  .summary-card {{
    font-size: 11px;
    color: #6e505f;
    background: rgba(255, 255, 255, 0.85);
    border-radius: 14px;
    padding: 10px 14px;
    line-height: 1.6;
  }}
  .today-count {{ font-size: 24px; color: #d98ba0; font-family: "Georgia", serif; font-weight: 600; }}
  .today-count-unit {{ font-size: 11px; color: #b08d98; margin-left: 3px; }}

  /* ---- 悬浮撤回菜单 (长按触发) ---- */
  #context-menu {{
    position: absolute;
    z-index: 9999;
    background: #ffffff;
    border-radius: 10px;
    box-shadow: 0 6px 16px rgba(0, 0, 0, 0.12);
    padding: 10px 20px;
    font-size: 14px;
    color: #d98ba0;
    cursor: pointer;
    font-weight: 500;
    transform-origin: center bottom;
    animation: popIn 0.2s cubic-bezier(0.175, 0.885, 0.32, 1.275);
  }}
  #context-menu:hover {{ background: #fdf6f8; }}
  @keyframes popIn {{ 0% {{ transform: scale(0.8); opacity: 0; }} 100% {{ transform: scale(1); opacity: 1; }} }}

</style>
</head>
<body>

<div id="app">

  <div id="nav">
    <a class="nav-icon" href="/" title="首页">✦</a>
    <a class="nav-icon" href="{diary_url}" title="日记">✎</a>
  </div>

  <div id="main">
    <div id="header">
      <img id="header-avatar" src="{CHAT_AVATAR_CHARON}" alt="Charon">
      <div id="header-text">
        <div class="brand">CHARON</div>
        <div class="sub"><span class="status-dot" id="status-dot"></span><span id="status-label">加载中…</span></div>
      </div>
      <div class="signature">@Seraphina</div>
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

<!-- 全局唯一的撤回弹窗容器 -->
<div id="context-menu" style="display: none;">撤回</div>

<script>
const CODE = {json.dumps(code_param)};
const AVATAR_CHARON = {json.dumps(CHAT_AVATAR_CHARON)};
const AVATAR_USER = {json.dumps(CHAT_AVATAR_USER)};
const messagesEl = document.getElementById('messages');
const inputEl = document.getElementById('input');
const sendBtn = document.getElementById('send-btn');
const panelBody = document.getElementById('panel-body');
const statusDot = document.getElementById('status-dot');
const statusLabel = document.getElementById('status-label');
const contextMenu = document.getElementById('context-menu');

let pressTimer;
let targetMsgId = null;
let targetMsgRow = null;

function apiUrl(path) {{
  const sep = path.includes('?') ? '&' : '?';
  return CODE ? `${{path}}${{sep}}code=${{encodeURIComponent(CODE)}}` : path;
}}

function formatTime(isoStr) {{
  if (!isoStr) return '';
  const d = new Date(isoStr);
  if (isNaN(d.getTime())) return '';
  const h = String(d.getHours()).padStart(2, '0');
  const m = String(d.getMinutes()).padStart(2, '0');
  return h + ':' + m;
}}

function renderMsgRow(role, content, createdAt, pending, msgId) {{
  const row = document.createElement('div');
  row.className = 'msg-row ' + (role === 'user' ? 'user' : 'charon');
  if (msgId) row.dataset.msgId = msgId;

  const avatar = document.createElement('img');
  avatar.className = 'msg-avatar';
  avatar.src = role === 'user' ? AVATAR_USER : AVATAR_CHARON;
  row.appendChild(avatar);

  const col = document.createElement('div');
  col.className = 'msg-col';

  const bubble = document.createElement('div');
  bubble.className = 'bubble ' + (role === 'user' ? 'user' : 'charon') + (pending ? ' pending' : '');
  bubble.textContent = content;
  col.appendChild(bubble);

  // 移除了原来的内联撤回按钮，只保留纯净的时间戳
  const timeEl = document.createElement('div');
  timeEl.className = 'msg-time';
  timeEl.textContent = formatTime(createdAt);
  col.appendChild(timeEl);

  row.appendChild(col);
  return row;
}}

/* ---- 长按触发撤回的核心交互逻辑 ---- */
messagesEl.addEventListener('pointerdown', (e) => {{
  const bubble = e.target.closest('.bubble.user');
  if (!bubble) return;
  
  const row = bubble.closest('.msg-row');
  const msgId = row.dataset.msgId;
  if (!msgId) return;

  // 按下 500ms 后触发菜单
  pressTimer = setTimeout(() => {{
    targetMsgId = msgId;
    targetMsgRow = row;
    
    // 定位菜单到点击位置的上方
    contextMenu.style.display = 'block';
    const menuWidth = contextMenu.offsetWidth;
    contextMenu.style.left = Math.max(10, e.pageX - menuWidth / 2) + 'px';
    contextMenu.style.top = (e.pageY - 50) + 'px';
    
    // 如果设备支持，触发微小震动反馈
    if (navigator.vibrate) navigator.vibrate(40);
  }}, 500);
}});

// 移动或松开即取消计时
messagesEl.addEventListener('pointerup', () => clearTimeout(pressTimer));
messagesEl.addEventListener('pointercancel', () => clearTimeout(pressTimer));
messagesEl.addEventListener('pointermove', () => clearTimeout(pressTimer)); // 简单移动即取消，防误触

// 点击其他地方关闭菜单
document.addEventListener('click', (e) => {{
  if (e.target !== contextMenu && !e.target.closest('.bubble.user')) {{
    contextMenu.style.display = 'none';
  }}
}});

// 执行撤回
contextMenu.addEventListener('click', async () => {{
  contextMenu.style.display = 'none';
  if (!targetMsgId || !targetMsgRow) return;
  if (!confirm('删掉这条消息？')) return;
  
  try {{
    const res = await fetch(apiUrl('/api/chat-delete'), {{
      method: 'POST',
      headers: {{ 'Content-Type': 'application/json' }},
      body: JSON.stringify({{ id: targetMsgId }})
    }});
    const data = await res.json();
    if (data.ok) {{
      targetMsgRow.remove();
    }} else {{
      alert('删除失败：' + (data.error || '未知错误'));
    }}
  }} catch (e) {{
    alert('网络错误，没删成');
  }}
}});

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
      messagesEl.innerHTML = '<div id="empty-hint">还没有聊过，长按你发的消息可以撤回喔</div>';
      return;
    }}
    data.messages.forEach(m => messagesEl.appendChild(renderMsgRow(m.role, m.content, m.created_at, false, m.id)));
    scrollToBottom();
  }} catch (e) {{
    messagesEl.innerHTML = '<div id="empty-hint">网络错误</div>';
  }}
}}

async function loadStatus() {{
  try {{
    const res = await fetch(apiUrl('/api/chat-status'));
    const data = await res.json();
    if (!data.ok) {{
      panelBody.innerHTML = '<div class="panel-empty">加载失败</div>';
      statusLabel.textContent = '未知';
      return;
    }}

    statusLabel.textContent = data.status_label || '在线';
    statusDot.className = 'status-dot' + (data.mood_score < 50 ? ' low' : '');

    const moodPct = Math.max(0, Math.min(100, data.mood_score));
    let html = '';
    html += '<div class="stat-block">';
    html += '<div class="stat-label"><span style="background: rgba(255,255,255,0.7); padding-right: 4px;">情绪值</span><span class="stat-value" style="background: rgba(255,255,255,0.7); padding-left: 4px;">' + data.mood_score + '/100</span></div>';
    html += '<div class="stat-bar-track"><div class="stat-bar-fill" style="width:' + moodPct + '%"></div></div>';
    html += '</div>';
    
    // 隐藏其他数据区域，保持极简（如果需要可以自己取消注释）
    /*
    html += '<div class="stat-block">';
    html += '<div class="stat-label"><span>上次互动</span></div>';
    html += '<div class="stat-value">' + formatHours(data.hours_since_last_event) + '</div>';
    html += '</div>';
    */
    
    html += '<div class="stat-block" style="margin-top: 20px;">';
    html += '<div class="stat-label"><span style="background: rgba(255,255,255,0.7); padding-right: 4px;">今日互动</span></div>';
    html += '<div><span class="today-count">' + data.today_interaction_count + '</span><span class="today-count-unit">次</span></div>';
    html += '</div>';
    
    if (data.last_thought) {{
      html += '<div class="stat-block" style="margin-top:24px;">';
      html += '<div class="panel-title">心里话</div>';
      html += '<div class="thought-card">' + escapeHtml(data.last_thought) + '</div></div>';
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

  const nowIso = new Date().toISOString();
  const userRow = renderMsgRow('user', text, nowIso, false);
  messagesEl.appendChild(userRow);
  const pendingRow = renderMsgRow('charon', '…', nowIso, true);
  messagesEl.appendChild(pendingRow);
  scrollToBottom();

  const pendingBubble = pendingRow.querySelector('.bubble');

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
      if (data.user_msg_id) userRow.dataset.msgId = data.user_msg_id;
      if (data.charon_msg_id) pendingRow.dataset.msgId = data.charon_msg_id;
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
