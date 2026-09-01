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
    table 直接是表名（events / diary / chat_messages / app_config）。
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
    这张表统一存 period/mood/window_summary/model_config 这几类"只有一份、整体覆盖"的配置。"""
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
    """插入一条event记录。"""
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
        "id": msg_id, "role": role, "content": content, "created_at": created_at or datetime.now().isoformat()
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


# ---- 经期与健康管理 ----
def load_period():
    return get_app_config("period", {"last_start": None, "is_active": False, "avg_cycle_days": 28, "avg_period_days": 5})


def save_period(data):
    set_app_config("period", data)


def get_period_context():
    p = load_period()
    is_active = p.get("is_active", False)
    last_start = p.get("last_start")

    if is_active:
        if last_start:
            try:
                day_index = (date.today() - date.fromisoformat(last_start)).days + 1
                return f"她现在的状态：生理期第{day_index}天，身体比较敏感，可能会累、怕冷、情绪波动，需要格外体贴关心。"
            except Exception:
                pass
        return "她现在的状态：处于生理期中，身体比较敏感，需要体贴。"
    else:
        if last_start:
            try:
                day_index = (date.today() - date.fromisoformat(last_start)).days + 1
                cycle = p.get("avg_cycle_days", 28)
                days_to_next = cycle - day_index
                if 0 <= days_to_next <= 3:
                    return f"距离她下次生理期大概还有{days_to_next}天，可以提前提醒她注意保暖别熬夜。"
            except Exception:
                pass
    return ""


# ---- 零摩擦情感收纳角 ----
def get_sticky_note():
    return get_app_config("sticky_note", {"content": "我在这里。有什么想说的，随时告诉我。", "updated_at": None})


def set_sticky_note(content):
    set_app_config("sticky_note", {"content": content, "updated_at": datetime.now().isoformat()})


def load_love_letters():
    """从 app_config 表的 love_letters 键中读取完整情书列表，无需新表，免去维护成本"""
    return get_app_config("love_letters", [])


def add_love_letter(content):
    """写情书直接打包成 JSON 列表，追加在现有的 app_config 中，最多保留 100 封"""
    letters = load_love_letters()
    letters.append({
        "id": new_msg_id(),
        "content": content,
        "created_at": datetime.now().isoformat(),
        "is_read": False
    })
    letters = letters[-100:]  # 保留最近 100 封信
    set_app_config("love_letters", letters)


# ---- 情绪值状态机 ----
MOOD_BASELINE = 50
MOOD_MAX = 100
MOOD_MIN = 0
MOOD_DECAY_PER_HOUR = 4
MOOD_RECOVERY_PER_EVENT = 8
MOOD_RECOVERY_MISS_YOU = 20


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
    mood = load_mood()
    mood["score"] = min(MOOD_MAX, mood["score"] + amount)
    mood["last_updated"] = datetime.now().isoformat()
    save_mood(mood)
    return mood["score"]


def get_mood_context(score, hours_gap):
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


# ---- 核心业务路由 ----

@app.route("/api/period/toggle", methods=["POST"])
def toggle_period():
    """一键切换生理期状态，专供移动端 Health 面板使用"""
    if not _check_chat_auth(request):
        return jsonify({"ok": False, "error": "unauthorized"}), 401
    
    p = load_period()
    is_active = p.get("is_active", False)
    
    if is_active:
        p["is_active"] = False  # 结束经期
    else:
        p["is_active"] = True  # 开始经期
        p["last_start"] = date.today().
