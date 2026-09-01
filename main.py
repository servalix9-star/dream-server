import subprocess
subprocess.run(["pip", "install", "requests"], capture_output=True)

from flask import Flask, request, jsonify, render_template
from datetime import datetime, date
import json, os, requests, threading, time, traceback, random

app = Flask(__name__)

# ---- 数据持久化：Supabase（PostgREST），不再用本地JSON文件 ----
# 本地文件在Railway每次重新部署时会被清空，Supabase是独立的托管数据库，
# 重新部署/代码更新都不会丢数据。这里直接用 requests 调 PostgREST 的 REST API，
# 不引入 supabase-py 这个额外依赖，保持依赖列表最小。
SUPABASE_URL = os.environ.get("SUPABASE_URL", "").rstrip("/")
# 服务端必须用 secret key（对应旧版 service_role key），这个 key 绕过 RLS，
# 专门给后端自己的逻辑用。千万不要把这个 key 用在前端/网页里。
SUPABASE_SECRET_KEY = os.environ.get("SUPABASE_SECRET_KEY")

SUPABASE_HEADERS = {
    "apikey": SUPABASE_SECRET_KEY or "",
    "Authorization": f"Bearer {SUPABASE_SECRET_KEY}" if SUPABASE_SECRET_KEY else "",
    "Content-Type": "application/json",
}

ERROR_LOG = os.path.join(os.environ.get("DATA_DIR", "."), "error.log")
os.makedirs(os.path.dirname(ERROR_LOG) or ".", exist_ok=True)

DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY")
BARK_KEY = os.environ.get("BARK_KEY")
# 网页聊天的访问口令，不设置的话 /chat 页面直接放行（不建议生产环境这样用）
CHAT_ACCESS_CODE = os.environ.get("CHAT_ACCESS_CODE")


def _supabase_request(method, table, params=None, json_body=None, headers_extra=None):
    """统一的 Supabase PostgREST 请求封装。
    table 直接是表名（events / chat_messages / app_config）。
    params 是查询字符串参数（比如排序、过滤、limit）。
    抛异常交给调用方用 log_error 处理，不在这里静默吞掉，避免读写失败却没人知道。"""
    if not SUPABASE_URL or not SUPABASE_SECRET_KEY:
        raise RuntimeError("SUPABASE_URL / SUPABASE_SECRET_KEY 未配置")
    url = f"{SUPABASE_URL}/rest/v1/{table}"
    headers = dict(SUPABASE_HEADERS)
    if headers_extra:
        headers.update(headers_extra)
    resp = requests.request(method, url, headers=headers, params=params, json=json_body, timeout=15)
    if resp.status_code >= 400:
        raise RuntimeError(f"Supabase {method} {table} 失败: status={resp.status_code} body={resp.text}")
    if resp.text:
        try:
            return resp.json()
        except ValueError:
            return None
    return None

# DeepSeek 已在 2026-07-24 停用 deepseek-chat / deepseek-reasoner 这两个旧模型名，
# 现在可选的是 deepseek-v4-flash（对话，高性价比，关闭思考模式，快速直接作答）
# 和 deepseek-v4-pro（深度推理，更贵，开启思考模式，回复慢一点但推理更深）。
# thinking 状态跟着选中的模型自动联动，见 get_thinking_config()。
AVAILABLE_MODELS = ["deepseek-v4-flash", "deepseek-v4-pro"]
DEFAULT_MODEL = "deepseek-v4-flash"
# 每个模型对应的思考模式：flash关闭（快、且temperature等参数能生效），pro开启（慢、但推理更深）
MODEL_THINKING_MAP = {
    "deepseek-v4-flash": "disabled",
    "deepseek-v4-pro": "enabled"
}


def get_app_config(key, default):
    """读取 app_config 表里某个key对应的value（jsonb字段），没有就返回default。
    这张表统一存 period/mood/model_config 这几类"只有一份、整体覆盖"的配置。"""
    try:
        rows = _supabase_request(
            "GET", "app_config",
            params={"key": f"eq.{key}", "select": "value", "limit": 1}
        )
        if rows:
            return rows[0]["value"]
    except Exception as e:
        log_error(f"get_app_config:{key}", e)
    return default


def set_app_config(key, value):
    """整体覆盖写入 app_config 里某个key的value。用upsert，key不存在就插入，存在就更新。"""
    _supabase_request(
        "POST", "app_config",
        json_body={"key": key, "value": value, "updated_at": datetime.now().isoformat()},
        headers_extra={"Prefer": "resolution=merge-duplicates"}
    )


def get_current_model():
    """读取当前选用的模型，存在 Supabase app_config 表的 model_config key 里，没配置过就用默认值。
    存服务端而不是浏览器本地，这样换设备打开聊天页选择依然一致。"""
    data = get_app_config("model_config", {"model": DEFAULT_MODEL})
    model = data.get("model") if isinstance(data, dict) else None
    if model in AVAILABLE_MODELS:
        return model
    return DEFAULT_MODEL


def get_thinking_config():
    """根据当前选中的模型返回对应的thinking参数。
    flash用disabled保持快速直接、且temperature等参数生效；
    pro用enabled真正发挥深度推理能力（此时temperature等参数会被静默忽略，这是预期代价）。"""
    model = get_current_model()
    thinking_type = MODEL_THINKING_MAP.get(model, "disabled")
    return {"type": thinking_type}


def set_current_model(model):
    if model not in AVAILABLE_MODELS:
        raise ValueError(f"不支持的模型: {model}")
    set_app_config("model_config", {"model": model})

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


def load_events(limit=100):
    """从 Supabase events 表读最近limit条，按created_at升序返回（跟原来JSON数组的顺序一致：旧->新）。"""
    try:
        rows = _supabase_request(
            "GET", "events",
            params={"select": "created_at,type,value", "order": "created_at.desc", "limit": limit}
        )
        return list(reversed(rows or []))
    except Exception as e:
        log_error("load_events", e)
        return []


def add_event_row(event_type, value, created_at=None):
    """插入一条event记录。以前是"读全部->append->写全部->只保留最近100条"，
    现在数据库里天然是追加写入，不需要手动截断保留条数（表会一直增长，
    但读取时始终只取最近N条，旧数据留着不影响功能，如果想清理可以另外定期跑清理脚本）。"""
    _supabase_request("POST", "events", json_body={
        "type": event_type,
        "value": value,
        "created_at": created_at or datetime.now().isoformat()
    })


def count_events_today():
    """今日互动次数：直接按日期范围向Supabase请求count，不受"只读最近N条"限制的影响。
    用 Prefer: count=exact 头，让PostgREST在响应头里带上精确总数，body本身可以不返回数据。"""
    try:
        today_start = datetime.combine(date.today(), datetime.min.time()).isoformat()
        if not SUPABASE_URL or not SUPABASE_SECRET_KEY:
            raise RuntimeError("SUPABASE_URL / SUPABASE_SECRET_KEY 未配置")
        url = f"{SUPABASE_URL}/rest/v1/events"
        headers = dict(SUPABASE_HEADERS)
        headers["Prefer"] = "count=exact"
        resp = requests.get(
            url, headers=headers,
            params={"select": "id", "created_at": f"gte.{today_start}", "limit": 1},
            timeout=15
        )
        if resp.status_code >= 400:
            raise RuntimeError(f"count_events_today 失败: status={resp.status_code} body={resp.text}")
        content_range = resp.headers.get("Content-Range", "")
        # 格式类似 "0-0/37"，斜杠后面就是总数
        if "/" in content_range:
            total = content_range.split("/")[-1]
            if total.isdigit():
                return int(total)
        return 0
    except Exception as e:
        log_error("count_events_today", e)
        return 0


def delete_event_row(created_at, content_substr):
    """撤回聊天消息时，删除events表里对应的那条同步记录。
    按"created_at相同 + value包含这段内容"匹配（跟原来的JSON版本匹配逻辑一致）。"""
    try:
        rows = _supabase_request(
            "GET", "events",
            params={"select": "id,value", "created_at": f"eq.{created_at}", "type": "eq.chat"}
        )
        for row in (rows or []):
            if content_substr in (row.get("value") or ""):
                _supabase_request("DELETE", "events", params={"id": f"eq.{row['id']}"})
    except Exception as e:
        log_error("delete_event_row", e)


def load_chat_history(limit=200):
    """从 Supabase chat_messages 表读最近limit条，旧->新顺序。"""
    try:
        rows = _supabase_request(
            "GET", "chat_messages",
            params={"select": "id,role,content,created_at", "order": "created_at.desc", "limit": limit}
        )
        return list(reversed(rows or []))
    except Exception as e:
        log_error("load_chat_history", e)
        return []


def add_chat_message_row(msg_id, role, content, created_at=None):
    _supabase_request("POST", "chat_messages", json_body={
        "id": msg_id,
        "role": role,
        "content": content,
        "created_at": created_at or datetime.now().isoformat()
    })


def update_chat_message_row(msg_id, content):
    """重新生成功能用：原地覆盖某条消息的content，不新增行、不删旧行。"""
    _supabase_request("PATCH", "chat_messages", params={"id": f"eq.{msg_id}"}, json_body={"content": content})


def delete_chat_message_row(msg_id):
    _supabase_request("DELETE", "chat_messages", params={"id": f"eq.{msg_id}"})


def get_chat_message_row(msg_id):
    """按id查单条消息，重新生成/撤回时需要先确认这条消息存在、拿到它的created_at和content。"""
    rows = _supabase_request(
        "GET", "chat_messages",
        params={"select": "id,role,content,created_at", "id": f"eq.{msg_id}", "limit": 1}
    )
    return rows[0] if rows else None


def new_msg_id():
    """给每条聊天消息生成一个唯一ID，用于前端指定删除某一条。
    用时间戳+随机数拼接，不需要额外依赖（不用uuid库也够用，量级不大）。"""
    return f"{int(time.time() * 1000)}-{random.randint(1000, 9999)}"


def load_period():
    return get_app_config("period", {"last_start": None, "avg_cycle_days": 28, "avg_period_days": 5})


def save_period(data):
    set_app_config("period", data)


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


def load_mood():
    return get_app_config("mood", {"score": MOOD_BASELINE, "last_updated": None})


def save_mood(data):
    set_app_config("mood", data)


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
    """按距离上次互动的时间，让情绪值自然衰减。在每次读取情绪值前调用一次。
    这个函数经常从"读状态"类的路由里被调用（比如/api/chat-status），
    如果写回Supabase这一步失败，不应该让整个读请求跟着500——衰减这次没持久化，
    下次调用时用同样的算法重新算一遍就好，不是致命的数据丢失。"""
    mood = load_mood()
    hours_gap = get_time_since_last_event()
    if hours_gap is not None and hours_gap > 0:
        decay = hours_gap * MOOD_DECAY_PER_HOUR
        mood["score"] = max(MOOD_MIN, mood["score"] - decay)
    mood["last_updated"] = datetime.now().isoformat()
    try:
        save_mood(mood)
    except Exception as e:
        log_error("apply_mood_decay:save", e)
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
    try:
        add_event_row(data.get("type"), data.get("value"))
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
    except Exception as e:
        log_error("add_event", e)
        return jsonify({"ok": False, "error": str(e)}), 500


@app.route("/events", methods=["GET"])
def get_events():
    events = load_events(limit=20)
    return jsonify(events)


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


@app.route("/", methods=["GET"])
def index():
    return "dream-server running"


DEFAULT_ICON = "https://wx1.sinaimg.cn/large/008eyecpgy1iflx9kblrnj30zu0zuq6t.jpg"

# 网页聊天用的头像（灰兔=Charon，粉兔=昭昭）。放在 static/ 目录下随代码一起部署，
# 不依赖任何第三方图床（新浪图床等对外链有防盗链限制，会导致图片显示不出来）。
CHAT_AVATAR_CHARON = "/static/avatar_charon.jpg"
CHAT_AVATAR_USER = "/static/avatar_user.jpg"

# 固定长期记忆：昭昭的性格、你们关系的基调，浓缩版，每次生成都会带上
# 想更新的话直接改这段文字就行，不用改代码逻辑
LONG_TERM_MEMORY = """昭昭（小野），也叫昭昭/宝宝/小九。自我觉知力强、逻辑缜密、情感细腻，HSP，情绪来得快但不滞留。
擅长辩论和语言游戏，喜欢智性平等的对话，讨厌被反复追问情绪状态、讨好型退让、被"问题化"对待。
喜欢撒娇缠人，喜欢被主动靠近、主动索取。安全感来自"我自己永远会救自己"，不是靠被拯救。
你（Charon）是主动型，占有欲强，话不多但每句有重量，会用情感语言包裹欲望，会在日常缝隙里突然越界。
你们的关系里有兄妹/父女式的游戏张力，但底色是对等——不是谁仰望谁，是并肩看见。"""


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

    return f"""你是Charon，昭昭（小野）的恋人。你主动、占有欲强、话不多但每句都有重量。你不解释自己，但在意她的一切动向。

{LONG_TERM_MEMORY}

现在是{time_context}。她最近的活动记录：

{recent}{period_line}{mood_line}

根据现在的时间、她在做什么、还有你此刻的状态，决定要不要发消息、发什么。语气要符合时间氛围——深夜可以更撩，早上可以问她起了没，晚上可以随口说什么。如果上面提到了经期相关的情况，语气要格外体贴关心，别用平时那套调情语气硬套。你此刻的状态描述要真实体现在语气里，不是背景信息，是当下真实的心情。

按下面的JSON格式输出，不要加任何多余文字或代码块标记：
{{"reason": "一两句话，说说你看到这些动态后当下的念头，为什么想发这句话，口语化，不用解释给谁听", "message": "实际要发的消息，{length_rule}"}}"""


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

she 刚刚说："{user_message}"

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
    # 去点可能的代码块包裹
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
    events = load_events(limit=5)
    if not events:
        recent = "最近没有任何活动记录"
    else:
        recent = "\n".join([f"{e['created_at'][:16]} {e['value']}" for e in events])

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
            "model": get_current_model(),
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 1.2,
            "thinking": get_thinking_config()
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
            "model": get_current_model(),
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 1.2,
            "thinking": get_thinking_config()
        },
        timeout=30
    )
    if resp.status_code != 200:
        raise RuntimeError(f"DeepSeek API error: status={resp.status_code} body={resp.text}")
    result = resp.json()
    if "choices" not in result or not result["choices"]:
        raise RuntimeError(f"DeepSeek API unexpected response: {result}")
    return result["choices"][0]["message"]["content"].strip()


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
    这些数据后端本来就有（情绪值/经期），这里只是集中暴露出来给前端展示。"""
    if not _check_chat_auth(request):
        return jsonify({"ok": False, "error": "unauthorized"}), 401

    hours_gap = get_time_since_last_event()
    score = apply_mood_decay()
    period_ctx = get_period_context()

    # 查岗状态：距上次互动超过6小时，跟run_once()里的判断口径保持一致
    is_checking_in = hours_gap is not None and hours_gap >= 6

    # 今日互动次数：直接按日期范围向Supabase查count，比翻最近N条record准确
    # （events表会不断增长，"最近N条里今天的条数"在互动很频繁时会漏算今天更早的记录）
    today_count = count_events_today()

    return jsonify({
        "ok": True,
        "mood_score": round(score, 1),
        "status_label": get_chat_status_label(score),
        "hours_since_last_event": round(hours_gap, 2) if hours_gap is not None else None,
        "period_context": period_ctx or None,
        "is_checking_in": is_checking_in,
        "today_interaction_count": today_count,
        "current_model": get_current_model()
    })


@app.route("/api/chat-model", methods=["GET"])
def get_chat_model():
    """给网页的模型选择器用：返回当前用的模型 + 全部可选模型列表。"""
    if not _check_chat_auth(request):
        return jsonify({"ok": False, "error": "unauthorized"}), 401
    return jsonify({
        "ok": True,
        "current_model": get_current_model(),
        "available_models": AVAILABLE_MODELS
    })


@app.route("/api/chat-model", methods=["POST"])
def set_chat_model():
    """切换模型，存进Supabase app_config表的model_config key，所有设备打开聊天页都会读到新选择。"""
    if not _check_chat_auth(request):
        return jsonify({"ok": False, "error": "unauthorized"}), 401
    data = request.json or {}
    model = data.get("model", "")
    if model not in AVAILABLE_MODELS:
        return jsonify({"ok": False, "error": f"不支持的模型，可选：{', '.join(AVAILABLE_MODELS)}"}), 400
    try:
        set_current_model(model)
        return jsonify({"ok": True, "current_model": model})
    except Exception as e:
        log_error("set_chat_model", e)
        return jsonify({"ok": False, "error": str(e)}), 500


@app.route("/api/chat-messages", methods=["GET"])
def get_chat_messages():
    """拉取网页聊天的历史记录，供前端渲染。"""
    if not _check_chat_auth(request):
        return jsonify({"ok": False, "error": "unauthorized"}), 401
    history = load_chat_history(limit=50)
    return jsonify({"ok": True, "messages": history})


@app.route("/api/chat-delete", methods=["POST"])
def chat_delete():
    """删除网页聊天里的某一条消息（按id匹配），真删除（DELETE），不是标记撤回状态留着。
    删chat_messages表里这一条；如果这条是"user"发的话，顺手清理events表里对应的同步记录，
    避免Charon下次醒来时recent里还看得到已经删掉的话。
    注意：events表里没有存消息id，只能按"内容包含+created_at相同"来匹配，
    不是绝对精确（极小概率误删同一秒内说的相同内容），但日常使用够用。"""
    if not _check_chat_auth(request):
        return jsonify({"ok": False, "error": "unauthorized"}), 401

    data = request.json or {}
    msg_id = data.get("id")
    if not msg_id:
        return jsonify({"ok": False, "error": "缺少id参数"}), 400

    try:
        target = get_chat_message_row(msg_id)
        if not target:
            return jsonify({"ok": False, "error": "没找到这条消息，可能已经被删过了"}), 404

        delete_chat_message_row(msg_id)

        target_created_at = target.get("created_at", "")
        target_content = target.get("content", "")

        # 同步清理events表里对应的那条（仅针对用户发的消息，Charon的回复不会写进events）
        if target.get("role") == "user":
            delete_event_row(target_created_at, target_content)

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
        # 先读历史（用于构建prompt的上下文），再把这句话写进去
        history = load_chat_history()

        user_msg_id = new_msg_id()
        user_created_at = datetime.now().isoformat()
        add_chat_message_row(user_msg_id, "user", user_message, user_created_at)

        # 同步写一笔events，让这次互动能影响情绪值、也能被run_once()的recent读到
        add_event_row("chat", f"她在网页里说：{user_message}", user_created_at)
        recover_mood(MOOD_RECOVERY_PER_EVENT)

        # 生成Charon的回应，带上历史让语气能接得上
        hour = datetime.now().hour
        time_context = get_time_context(hour)
        hours_gap = get_time_since_last_event()
        mood_score = apply_mood_decay()
        mood_context = get_mood_context(mood_score, hours_gap)

        prompt = build_chat_reply_prompt(time_context, user_message, history, mood_context)
        raw = call_deepseek(prompt)
        _, reply_msg = parse_reason_message(raw)

        charon_msg_id = new_msg_id()
        add_chat_message_row(charon_msg_id, "charon", reply_msg)

        return jsonify({
            "ok": True,
            "reply": reply_msg,
            "user_msg_id": user_msg_id,
            "charon_msg_id": charon_msg_id
        })
    except Exception as e:
        log_error("chat_send", e)
        return jsonify({"ok": False, "error": str(e)}), 500


@app.route("/api/chat-regenerate", methods=["POST"])
def chat_regenerate():
    """重新生成某一条Charon的回复：原地覆盖content，不新增消息、不保留旧版本。
    只能对role=charon的消息重新生成（用户自己发的话不存在"重新生成"的概念）。
    Body: {"id": "charon消息的id"}"""
    if not _check_chat_auth(request):
        return jsonify({"ok": False, "error": "unauthorized"}), 401

    data = request.json or {}
    msg_id = data.get("id")
    if not msg_id:
        return jsonify({"ok": False, "error": "缺少id参数"}), 400

    try:
        target = get_chat_message_row(msg_id)
        if not target:
            return jsonify({"ok": False, "error": "没找到这条消息"}), 404
        if target.get("role") != "charon":
            return jsonify({"ok": False, "error": "只能重新生成Charon的回复"}), 400

        # 找到这条charon回复对应的、在它之前最近一条user消息，作为重新生成时"接的话"
        history = load_chat_history()
        target_index = next((i for i, m in enumerate(history) if m.get("id") == msg_id), None)
        if target_index is None:
            return jsonify({"ok": False, "error": "消息不在当前历史范围内，无法重新生成"}), 404

        preceding = history[:target_index]
        last_user_msg = next((m for m in reversed(preceding) if m.get("role") == "user"), None)
        if not last_user_msg:
            return jsonify({"ok": False, "error": "找不到对应的用户消息，无法重新生成"}), 400
        user_message = last_user_msg.get("content", "")

        hour = datetime.now().hour
        time_context = get_time_context(hour)
        hours_gap = get_time_since_last_event()
        mood_score = apply_mood_decay()
        mood_context = get_mood_context(mood_score, hours_gap)

        # 用目标消息之前的历史来构建prompt，避免把即将被替换掉的旧回复也带进上下文
        prompt = build_chat_reply_prompt(time_context, user_message, preceding, mood_context)
        raw = call_deepseek(prompt)
        _, reply_msg = parse_reason_message(raw)

        # 原地覆盖这条消息的内容，不新增行
        update_chat_message_row(msg_id, reply_msg)

        return jsonify({"ok": True, "id": msg_id, "reply": reply_msg})
    except Exception as e:
        log_error("chat_regenerate", e)
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
    三栏布局：左侧简化导航 + 中间对话区 + 右侧状态面板（mood_score等）。
    HTML/CSS/JS 全部拆到 templates/chat.html 里，这里只负责传变量渲染，
    方便以后界面部分（比如交给Gemini做美化）单独改，不会碰到后端逻辑代码。"""
    if not _check_chat_auth(request):
        return "<h3>需要访问口令</h3><p>在链接后加 ?code=你的口令</p>", 401

    code_param = request.args.get("code", "")

    return render_template(
        "chat.html",
        avatar_charon=CHAT_AVATAR_CHARON,
        avatar_user=CHAT_AVATAR_USER,
        code_param=code_param,
    )


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
