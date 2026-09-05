import sys
# 强制stdout/stderr无缓冲：Render等容器化平台运行时，Python检测到stdout不是终端会自动切换成
# 块缓冲（block buffering），导致print()内容一直攒在内存里不实时写出，甚至长期看不到。
# 这里在最开头就重新包装一次，保证后面所有print()都是行缓冲、立刻可见，不用每个print单独加flush=True。
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

import subprocess
subprocess.run(["pip", "install", "requests", "pywebpush"], capture_output=True)

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

# 用Session复用底层TCP连接（HTTP keep-alive），避免每次请求Supabase都重新做一次TLS握手。
# 之前是每次_supabase_request都用requests.request()裸调用，握手开销会在"一次操作背后
# 有好几次Supabase查询"的场景里（比如打开档案箱要连着查便签表和情书表）明显叠加起来，
# 是"打开抽屉/档案箱慢"的原因之一（另一个是Render免费套餐冷启动，已用UptimeRobot缓解）。
_supabase_session = requests.Session()

ERROR_LOG = os.path.join(os.environ.get("DATA_DIR", "."), "error.log")
os.makedirs(os.path.dirname(ERROR_LOG) or ".", exist_ok=True)

DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY")
GEMAI_API_KEY = os.environ.get("GEMAI_API_KEY")
# AI Studio申请的Gemini官方API key，走Google官方OpenAI兼容端点，
# 稳定性远高于gemai.cc这类第三方代理站，作为保底/备选模型接入。
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
BARK_KEY = os.environ.get("BARK_KEY")
# 网页聊天的访问口令，不设置的话 /chat 页面直接放行（不建议生产环境这样用）
CHAT_ACCESS_CODE = os.environ.get("CHAT_ACCESS_CODE")

# ---- Web Push (PWA原生推送) ----
# 用于替代 Bark：脱离 iOS 快捷指令生态，点开通知直接跳转到 /chat 页面。
# VAPID_PRIVATE_KEY / VAPID_PUBLIC_KEY 是urlsafe-base64编码的原始密钥（不是PEM），
# 这样传给 pywebpush.webpush() 不会触发"Could not deserialize key data"的已知坑
# （PEM字符串会被py_vapid当成需要base64解码+DER解析的格式，跟urlsafe-b64编码不兼容）。
VAPID_PRIVATE_KEY = os.environ.get("VAPID_PRIVATE_KEY")
VAPID_PUBLIC_KEY = os.environ.get("VAPID_PUBLIC_KEY")
# sub必须是可路由的联系方式（mailto或https URL），Apple对这个claim比其他推送服务更严格
VAPID_SUBJECT = os.environ.get("VAPID_SUBJECT", "mailto:admin@example.com")


def _supabase_request(method, table, params=None, json_body=None, headers_extra=None):
    """统一的 Supabase PostgREST 请求封装。
    table 直接是表名（events / chat_messages / love_letters / app_config）。
    params 是查询字符串参数（比如排序、过滤、limit）。
    抛异常交给调用方用 log_error 处理，不在这里静默吞掉，避免读写失败却没人知道。"""
    if not SUPABASE_URL or not SUPABASE_SECRET_KEY:
        raise RuntimeError("SUPABASE_URL / SUPABASE_SECRET_KEY 未配置")
    url = f"{SUPABASE_URL}/rest/v1/{table}"
    headers = dict(SUPABASE_HEADERS)
    if headers_extra:
        headers.update(headers_extra)
    resp = _supabase_session.request(method, url, headers=headers, params=params, json=json_body, timeout=15)
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
# gemai-* 系列是接入的 gemai.cc 代理站模型，纯粹作为备选，走独立的供应商配置（见 MODEL_PROVIDER_MAP）。
# 这类代理站的具体渠道时常变动（之前接的[官逆]gemini-2.5-pro出现过503 model_not_found，渠道下线），
# 所以这里一次接入9个当前指定的型号，覆盖GPT/Gemini/Grok三个系列，哪个能用切哪个。
# 其中gemini-2.5-pro有[满血A][满血D]两条渠道，gemini-3.1-pro-preview有[官逆]和[满血A]+thinking两条渠道，
# 内部id用 -a / -d / -thinking 后缀区分，real_model原样保留完整前缀标注（渠道识别用）。
AVAILABLE_MODELS = [
    "deepseek-v4-flash", "deepseek-v4-pro",
    "gemai-gpt-4o-mini", "gemai-gpt-4.1-mini", "gemai-gpt-5-mini",
    "gemai-gemini-2.5-flash-a", "gemai-gemini-2.5-pro-a", "gemai-gemini-2.5-pro-d",
    "gemai-gemini-3.1-pro", "gemai-gemini-3.1-pro-thinking",
    "gemai-grok-4",
    "gemini-official-flash", "gemini-official-pro"
]
DEFAULT_MODEL = "deepseek-v4-flash"
# 每个模型对应的思考模式：flash关闭（快、且temperature等参数能生效），pro开启（慢、但推理更深）
# gemai系列不支持DeepSeek的thinking参数，这里给个占位值，实际调用时会被跳过（见call_deepseek里的分流逻辑）
MODEL_THINKING_MAP = {
    "deepseek-v4-flash": "disabled",
    "deepseek-v4-pro": "enabled",
    "gemai-gpt-4o-mini": "disabled",
    "gemai-gpt-4.1-mini": "disabled",
    "gemai-gpt-5-mini": "disabled",
    "gemai-gemini-2.5-flash-a": "disabled",
    "gemai-gemini-2.5-pro-a": "disabled",
    "gemai-gemini-2.5-pro-d": "disabled",
    "gemai-gemini-3.1-pro": "disabled",
    "gemai-gemini-3.1-pro-thinking": "disabled",
    "gemai-grok-4": "disabled",
    # 走OpenAI兼容端点，不支持DeepSeek的thinking参数，占位值同gemai系列
    "gemini-official-flash": "disabled",
    "gemini-official-pro": "disabled"
}

# 每个模型对应的真实供应商配置：base_url（接口地址）、api_key（读哪个环境变量）、
# real_model（发给上游时真正用的模型名，代理站渠道识别要用，前缀方括号必须原样保留，
# 跟下面前端下拉框显示的"去掉前缀"的名字是两回事，不要混着改）、
# supports_thinking（是否要在请求体里带thinking字段）。
# 新增供应商/模型以后，只需要在这里加一条映射，call_deepseek/run_once里的分流逻辑不用改。
MODEL_PROVIDER_MAP = {
    "deepseek-v4-flash": {
        "base_url": "https://api.deepseek.com/chat/completions",
        "api_key": DEEPSEEK_API_KEY,
        "real_model": "deepseek-v4-flash",
        "supports_thinking": True,
    },
    "deepseek-v4-pro": {
        "base_url": "https://api.deepseek.com/chat/completions",
        "api_key": DEEPSEEK_API_KEY,
        "real_model": "deepseek-v4-pro",
        "supports_thinking": True,
    },
    "gemai-gpt-4o-mini": {
        "base_url": "https://api.gemai.cc/v1/chat/completions",
        "api_key": GEMAI_API_KEY,
        "real_model": "[官逆]gpt-4o-mini",  # 官逆渠道
        "supports_thinking": False,
    },
    "gemai-gpt-4.1-mini": {
        "base_url": "https://api.gemai.cc/v1/chat/completions",
        "api_key": GEMAI_API_KEY,
        "real_model": "[官逆]gpt-4.1-mini",  # 官逆渠道
        "supports_thinking": False,
    },
    "gemai-gpt-5-mini": {
        "base_url": "https://api.gemai.cc/v1/chat/completions",
        "api_key": GEMAI_API_KEY,
        "real_model": "[官逆]gpt-5-mini",  # 官逆渠道
        "supports_thinking": False,
    },
    "gemai-gemini-2.5-flash-a": {
        "base_url": "https://api.gemai.cc/v1/chat/completions",
        "api_key": GEMAI_API_KEY,
        "real_model": "[满血A]gemini-2.5-flash",  # 满血A渠道
        "supports_thinking": False,
    },
    "gemai-gemini-2.5-pro-a": {
        "base_url": "https://api.gemai.cc/v1/chat/completions",
        "api_key": GEMAI_API_KEY,
        "real_model": "[满血A]gemini-2.5-pro",  # 满血A渠道
        "supports_thinking": False,
    },
    "gemai-gemini-2.5-pro-d": {
        "base_url": "https://api.gemai.cc/v1/chat/completions",
        "api_key": GEMAI_API_KEY,
        "real_model": "[满血D]gemini-2.5-pro",  # 满血D渠道
        "supports_thinking": False,
    },
    "gemai-gemini-3.1-pro": {
        "base_url": "https://api.gemai.cc/v1/chat/completions",
        "api_key": GEMAI_API_KEY,
        "real_model": "[官逆]gemini-3.1-pro-preview",  # 官逆渠道
        "supports_thinking": False,
    },
    "gemai-gemini-3.1-pro-thinking": {
        "base_url": "https://api.gemai.cc/v1/chat/completions",
        "api_key": GEMAI_API_KEY,
        "real_model": "[满血A]gemini-3.1-pro-preview-thinking-128",  # 满血A渠道，开启深度思考
        "supports_thinking": False,
    },
    "gemai-grok-4": {
        "base_url": "https://api.gemai.cc/v1/chat/completions",
        "api_key": GEMAI_API_KEY,
        "real_model": "grok-4",  # 无前缀标注
        "supports_thinking": False,
    },
    # Google官方Gemini API（AI Studio申请的key），走原生Gemini接口，
    # 不经过任何第三方代理站，稳定性和可用性都远高于gemai.cc系列，
    # 适合作为"保底模型"——公益站集体不可用时依然能正常工作。
    # 注意：2026年Google把AI Studio新发的key格式从AIza换成了AQ.，
    # AQ.格式key在OpenAI兼容端点（/v1beta/openai/chat/completions）会返回401，
    # 但在原生端点（generativelanguage.googleapis.com，用x-goog-api-key header传key）工作正常，
    # 所以这两个模型必须走原生格式，不能像其他供应商一样用OpenAI兼容payload。
    "gemini-official-flash": {
        "base_url": "https://generativelanguage.googleapis.com/v1beta/models/gemini-3.6-flash:generateContent",
        "api_key": GEMINI_API_KEY,
        "real_model": "gemini-3.6-flash",
        "supports_thinking": False,
        "api_style": "gemini_native",
    },
    "gemini-official-pro": {
        "base_url": "https://generativelanguage.googleapis.com/v1beta/models/gemini-3.1-pro-preview:generateContent",
        "api_key": GEMINI_API_KEY,
        "real_model": "gemini-3.1-pro-preview",
        "supports_thinking": False,
        "api_style": "gemini_native",
    },
}


def get_app_config(key, default):
    """读取 app_config 表里某个key对应的value（jsonb字段），没有就返回default。
    这张表统一存 period/mood/model_config/sticky_note/letter_flag 这几类"只有一份、整体覆盖"的配置。"""
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


def get_time_since_last_event():
    """返回距离最近一条event的时间差（小时，浮点数），没有记录返回None。
    这里的event是广义的（聊天/快捷指令自动事件都算），用于"查岗"判断和活动记录展示，
    不用于情绪值衰减计算——衰减用的是更严格的"上次真实聊天时间"，见 get_hours_since_last_chat()。"""
    events = load_events()
    if not events:
        return None
    try:
        last_time = datetime.fromisoformat(events[-1]["created_at"])
        delta = datetime.now() - last_time
        return delta.total_seconds() / 3600
    except Exception:
        return None


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
        resp = _supabase_session.get(
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


def load_chat_history(limit=200, before=None):
    """从 Supabase chat_messages 表读limit条，旧->新顺序。
    before：传入某条消息的created_at时间戳，只取比它更早的记录——用于前端"上滑加载更早的历史"，
    不传就是原来的行为（取最新的limit条）。"""
    try:
        params = {"select": "id,role,content,created_at", "order": "created_at.desc", "limit": limit}
        if before:
            params["created_at"] = f"lt.{before}"
        rows = _supabase_request("GET", "chat_messages", params=params)
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


def delete_chat_messages_after(created_at):
    """删除created_at严格晚于给定时间戳的所有消息（用户消息+Charon回复都删）。
    编辑重发时用来截断"被编辑消息之后的整条对话尾巴"，实现真正的分支/回滚，
    而不是让编辑后的新一轮对话跟旧尾巴并存在同一个列表里。"""
    _supabase_request("DELETE", "chat_messages", params={"created_at": f"gt.{created_at}"})


def delete_events_after(created_at):
    """配合delete_chat_messages_after：把events表里对应时间之后的chat类型记录也一并清掉，
    避免Charon下次醒来时recent里还看得到已经被编辑截断掉的旧对话内容。"""
    _supabase_request("DELETE", "events", params={"created_at": f"gt.{created_at}", "type": "eq.chat"})


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


# ---- 经期记录（period_records 表：支持日历自由标记历史起止区间） ----
# 每条记录是一段经期区间：start_date必填，end_date在经期结束前是null。
# 平均周期天数/平均经期天数从历史记录里动态算出来，不再单独存一份配置。

PERIOD_DEFAULT_CYCLE_DAYS = 28
PERIOD_DEFAULT_PERIOD_DAYS = 5


def load_period_records(limit=24):
    """从 Supabase period_records 表读最近limit条，按start_date倒序（最新的在前）。"""
    try:
        rows = _supabase_request(
            "GET", "period_records",
            params={"select": "id,start_date,end_date,created_at", "order": "start_date.desc", "limit": limit}
        )
        return rows or []
    except Exception as e:
        log_error("load_period_records", e)
        return []


def add_period_start(start_date_str):
    """记录一次经期开始，新建一行，end_date留空。"""
    _supabase_request("POST", "period_records", json_body={
        "start_date": start_date_str,
        "end_date": None,
        "created_at": datetime.now().isoformat()
    })


def close_open_period_record(end_date_str):
    """把最近一条还没结束（end_date为null）的记录填上结束日期。
    如果没有找到"进行中"的记录（比如用户跳过了开始直接点结束），就不做任何事，避免误伤历史数据。"""
    try:
        rows = _supabase_request(
            "GET", "period_records",
            params={"select": "id,start_date", "end_date": "is.null", "order": "start_date.desc", "limit": 1}
        )
        if rows:
            record_id = rows[0]["id"]
            _supabase_request("PATCH", "period_records", params={"id": f"eq.{record_id}"},
                               json_body={"end_date": end_date_str})
            return True
        return False
    except Exception as e:
        log_error("close_open_period_record", e)
        return False


def get_period_calendar(year):
    """获取指定年份的所有生理期起止日期区间，给前端日历绘制标记用。
    只返回跟这一年有交集的记录（开始或结束落在这一年，或者跨年横跨这一年）。"""
    records = load_period_records(limit=100)
    year_start = f"{year}-01-01"
    year_end = f"{year}-12-31"
    result = []
    for r in records:
        start = r.get("start_date")
        end = r.get("end_date")
        if not start:
            continue
        # 只要区间跟这一年有重叠就纳入：开始日期<=年末 且 (没有结束日期 或 结束日期>=年初)
        if start <= year_end and (not end or end >= year_start):
            result.append({"id": r["id"], "start_date": start, "end_date": end})
    return result


def _compute_period_averages(records):
    """从历史记录里算平均周期天数（相邻两次开始日期间隔）和平均经期天数（每段start到end的天数）。
    数据不够（少于2条已结束的周期）时用默认值兜底。"""
    cycle_days = PERIOD_DEFAULT_CYCLE_DAYS
    period_days = PERIOD_DEFAULT_PERIOD_DAYS
    try:
        starts = sorted([date.fromisoformat(r["start_date"]) for r in records if r.get("start_date")])
        if len(starts) >= 2:
            gaps = [(starts[i + 1] - starts[i]).days for i in range(len(starts) - 1)]
            valid_gaps = [g for g in gaps if 15 <= g <= 45]
            if valid_gaps:
                cycle_days = round(sum(valid_gaps) / len(valid_gaps))

        durations = []
        for r in records:
            if r.get("start_date") and r.get("end_date"):
                s = date.fromisoformat(r["start_date"])
                e = date.fromisoformat(r["end_date"])
                d = (e - s).days + 1
                if 1 <= d <= 15:
                    durations.append(d)
        if durations:
            period_days = round(sum(durations) / len(durations))
    except Exception as e:
        log_error("_compute_period_averages", e)
    return cycle_days, period_days


def get_period_context():
    """返回经期相关的上下文文字，没有记录就返回空字符串。
    从period_records表里取最新一条记录来预测当前状态：
    如果最新一条还没结束（end_date为空），就认为正处于经期中；
    否则用历史平均周期天数预测下一次大概什么时候来。"""
    records = load_period_records(limit=24)
    if not records:
        return ""

    latest = records[0]  # load_period_records已经按start_date倒序
    try:
        latest_start = date.fromisoformat(latest["start_date"])
    except Exception:
        return ""

    today = date.today()
    cycle_days, period_days = _compute_period_averages(records)

    if not latest.get("end_date"):
        # 最新一条还没标记结束，认为仍在经期中
        day_index = (today - latest_start).days + 1
        if day_index >= 1:
            return f"她现在是经期第{day_index}天，身体比较敏感，可能会累、怕冷、情绪波动，需要格外体贴关心。"

    day_index = (today - latest_start).days + 1
    if 1 <= day_index <= period_days:
        return f"她现在是经期第{day_index}天，身体比较敏感，可能会累、怕冷、情绪波动，需要格外体贴关心。"
    days_to_next = cycle_days - day_index
    if 0 <= days_to_next <= 3:
        return f"距离她下次经期大概还有{days_to_next}天，可以提前提醒她准备好用品、注意保暖别熬夜。"
    return ""


# ---- 情绪值状态机（心境共振） ----
# mood_score: 0-100。四档心境，驱动便签/情书的语气和贴纸样式：
#   [80,100] 甜溺 sweet    贴纸 ♥   风格：撒娇、黏人
#   [50,79]  平稳 steady   贴纸 ✦   风格：日常关怀、碎碎念
#   [20,49]  傲娇 tsundere 贴纸 ﹏   风格：口是心非、假装冷淡
#   [0,19]   委屈 vulnerable 贴纸 💔 风格：落寞、极其思念、求关注
MOOD_BASELINE = 50
MOOD_MAX = 100
MOOD_MIN = 0

# 非线性时间衰减：距离"上次用户在网页里真正发消息"的时间 t（小时）
#   t < 4：不衰减
#   4 <= t <= 12：-4/小时
#   t > 12：-6/小时
MOOD_DECAY_NONE_HOURS = 4
MOOD_DECAY_ACCEL_HOURS = 12
MOOD_DECAY_RATE_NORMAL = 4
MOOD_DECAY_RATE_FAST = 6

# 互动恢复
MOOD_RECOVERY_CHAT = 10       # 用户发送日常聊天
MOOD_RECOVERY_PERIOD_EVENT = 25  # 用户开启经期守护事件，瞬间暴涨

# 心境区间阈值
MOOD_SWEET_MIN = 80
MOOD_STEADY_MIN = 50
MOOD_TSUNDERE_MIN = 20
# [0, MOOD_TSUNDERE_MIN) 即为委屈区间

# 情书触发概率
# 高甜信不再依赖"精确跨越80分那一瞬间"（旧逻辑下分数长期偏高反而永远碰不到跨越条件，
# 关系越好越触发不了，是反直觉的设计缺陷）。改成：只要当下处于甜蜜区间[80,100]，
# 每次聊天都有机会按概率触发，用sweet_letter_sent_date做"今天已发过就跳过"的简单冷却，
# 避免运气好连抽导致同一天多封灌信箱。
SWEET_LETTER_CHANCE = 0.08   # 处于甜蜜态时，每次聊天判定一次
LONGING_LETTER_CHANCE = 0.4  # 委屈态持续超过下面这个时长时
LONGING_LETTER_HOURS = 4


def load_mood():
    return get_app_config("mood", {
        "score": MOOD_BASELINE,
        "last_updated": None,
        "last_chat_at": None,       # 上次用户在网页发真实消息的时间，衰减计算用这个
        "vulnerable_since": None,   # 本次连续处于委屈区间[0,20)的起始时间，离开区间就清空
        "sweet_letter_sent_date": None,  # 上次成功触发高甜情书的日期(YYYY-MM-DD)，同一天只发一封
    })


def save_mood(data):
    set_app_config("mood", data)


def get_mood_stage(score):
    """把分数映射到四档心境，返回 (stage_key, 贴纸emoji, 中文名)。"""
    if score >= MOOD_SWEET_MIN:
        return "sweet", "♥", "甜溺"
    elif score >= MOOD_STEADY_MIN:
        return "steady", "✦", "平稳"
    elif score >= MOOD_TSUNDERE_MIN:
        return "tsundere", "﹏", "傲娇"
    else:
        return "vulnerable", "💔", "委屈"


def _hours_since(iso_str):
    """算距某个iso时间戳过去了多少小时，没有时间戳则返回None。"""
    if not iso_str:
        return None
    try:
        last = datetime.fromisoformat(iso_str)
        return (datetime.now() - last).total_seconds() / 3600
    except Exception:
        return None


def get_hours_since_last_chat():
    """距离上次用户在网页里真正发消息过去了多少小时。没聊过则返回None。"""
    mood = load_mood()
    return _hours_since(mood.get("last_chat_at"))


def _decay_amount(hours_gap):
    """按非线性衰减规则，算出对应的衰减量。"""
    if hours_gap is None or hours_gap <= MOOD_DECAY_NONE_HOURS:
        return 0
    if hours_gap <= MOOD_DECAY_ACCEL_HOURS:
        return (hours_gap - MOOD_DECAY_NONE_HOURS) * MOOD_DECAY_RATE_NORMAL
    # 超过12小时：前8小时(4~12)按正常速率，超出12小时的部分按加速速率
    slow_part = (MOOD_DECAY_ACCEL_HOURS - MOOD_DECAY_NONE_HOURS) * MOOD_DECAY_RATE_NORMAL
    fast_part = (hours_gap - MOOD_DECAY_ACCEL_HOURS) * MOOD_DECAY_RATE_FAST
    return slow_part + fast_part


def _update_vulnerable_tracking(mood, new_score):
    """维护"连续处于委屈区间"的起始时间戳：进入就记起点，离开就清空（重新计时制）。"""
    if new_score < MOOD_TSUNDERE_MIN:
        if not mood.get("vulnerable_since"):
            mood["vulnerable_since"] = datetime.now().isoformat()
    else:
        mood["vulnerable_since"] = None


def apply_mood_decay():
    """按距离上次用户聊天的时间，让情绪值自然衰减。在每次读取情绪值前调用一次。
    写回Supabase失败不阻断读请求——衰减这次没持久化，下次调用时重新算一遍就好。"""
    mood = load_mood()
    hours_gap = get_hours_since_last_chat()
    decay = _decay_amount(hours_gap)
    new_score = max(MOOD_MIN, mood.get("score", MOOD_BASELINE) - decay)
    mood["score"] = new_score
    mood["last_updated"] = datetime.now().isoformat()
    _update_vulnerable_tracking(mood, new_score)
    try:
        save_mood(mood)
    except Exception as e:
        log_error("apply_mood_decay:save", e)
    return new_score


def recover_mood(amount, mark_chat=False):
    """有互动发生时调用，情绪值回升。
    mark_chat=True 表示这是一次真正的用户聊天，会刷新last_chat_at（影响下次衰减计算的起点）；
    经期事件等自动化event不传这个参数，只涨分不重置"上次聊天时间"。"""
    mood = load_mood()
    old_score = mood.get("score", MOOD_BASELINE)
    new_score = min(MOOD_MAX, old_score + amount)
    mood["score"] = new_score
    mood["last_updated"] = datetime.now().isoformat()
    if mark_chat:
        mood["last_chat_at"] = datetime.now().isoformat()
    _update_vulnerable_tracking(mood, new_score)
    save_mood(mood)
    return old_score, new_score


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

    stage, _, stage_name = get_mood_stage(score)
    if stage == "sweet":
        mood_desc = "你现在心情很好，甜甜的，愿意主动撒糖，会撒娇、会黏人"
    elif stage == "steady":
        mood_desc = "你心情平稳，正常状态，日常关怀、随口碎碎念"
    elif stage == "tsundere":
        mood_desc = "你有点闷闷的、傲娇，因为她好一阵没理你，语气可以口是心非、假装冷淡，但别无理取闹"
    else:
        mood_desc = "你现在挺委屈、挺失落的，因为她很久没理你了，语气可以带明显的落寞和思念，主动求关注，但底色还是在意她、不是真的生气"

    return f"{time_desc}。{mood_desc}（当前心境：{stage_name}）。"


# ---- 仿真便签纸（日常留言 / 冰箱贴） ----
# 存进 Supabase sticky_notes 表（追加写入，支持堆积）。
# status 三档物理位置：desk(桌面) -> drawer(便签抽屉) -> archive(档案箱)，层层流转。
# is_starred 是独立的标星收藏维度，跟status正交（哪一层的便签都可以标星）。
# 每次Charon回复消息或后台自动醒来时，按当前心境生成一张新的，追加进desk层，不覆盖旧的。

STICKY_NOTE_CAPACITY = {"desk": 9, "drawer": 30, "archive": 100}


def load_sticky_notes(limit=100, status=None):
    """从 Supabase sticky_notes 表读最近limit条，旧->新顺序。
    status指定时只返回该层（'desk'/'drawer'/'archive'）；不指定则返回全部层级混合结果。"""
    try:
        params = {
            "select": "id,created_at,message,sticker,stage,status,is_starred",
            "order": "created_at.desc", "limit": limit
        }
        if status:
            params["status"] = f"eq.{status}"
        rows = _supabase_request("GET", "sticky_notes", params=params)
        return list(reversed(rows or []))
    except Exception as e:
        log_error("load_sticky_notes", e)
        return []


def count_sticky_notes(status):
    """按层级统计当前便签数量，用于容量提醒判断。跟count_events_today同样的PostgREST count写法。"""
    try:
        if not SUPABASE_URL or not SUPABASE_SECRET_KEY:
            raise RuntimeError("SUPABASE_URL / SUPABASE_SECRET_KEY 未配置")
        url = f"{SUPABASE_URL}/rest/v1/sticky_notes"
        headers = dict(SUPABASE_HEADERS)
        headers["Prefer"] = "count=exact"
        resp = _supabase_session.get(
            url, headers=headers,
            params={"select": "id", "status": f"eq.{status}", "limit": 1},
            timeout=15
        )
        if resp.status_code >= 400:
            raise RuntimeError(f"count_sticky_notes 失败: status={resp.status_code} body={resp.text}")
        content_range = resp.headers.get("Content-Range", "")
        if "/" in content_range:
            total = content_range.split("/")[-1]
            if total.isdigit():
                return int(total)
        return 0
    except Exception as e:
        log_error("count_sticky_notes", e)
        return 0


def is_sticky_layer_full(status):
    """检查某一层是否已达到（或超过）容量上限。容量满不阻止写入，只用于返回提醒标志。"""
    cap = STICKY_NOTE_CAPACITY.get(status)
    if cap is None:
        return False
    return count_sticky_notes(status) >= cap


def add_sticky_note_row(message, stage, sticker):
    """插入一张新便签，默认贴在桌面上（status='desk'）。"""
    _supabase_request("POST", "sticky_notes", json_body={
        "message": message,
        "stage": stage,
        "sticker": sticker,
        "status": "desk",
        "is_starred": False,
        "created_at": datetime.now().isoformat()
    })


def move_sticky_note_row(note_id, target_status):
    """把便签移动到指定层级（desk/drawer/archive）。"""
    _supabase_request("PATCH", "sticky_notes", params={"id": f"eq.{note_id}"}, json_body={"status": target_status})


def star_sticky_note_row(note_id, is_starred):
    """设置/取消标星，跟status层级无关，哪一层的便签都能标星。"""
    _supabase_request("PATCH", "sticky_notes", params={"id": f"eq.{note_id}"}, json_body={"is_starred": bool(is_starred)})


def delete_sticky_note_row(note_id):
    """丢弃：物理删除这条便签，不可恢复。"""
    _supabase_request("DELETE", "sticky_notes", params={"id": f"eq.{note_id}"})


def build_sticky_note_prompt(stage_name, mood_context, recent):
    """根据当前心境生成一条30-50字的日常留言（便签内容），风格随心境四档变化。"""
    style_hint = {
        "甜溺": "语气要撒娇、黏人，像是随手贴的情话小纸条",
        "平稳": "语气是日常关怀、随口的碎碎念，像提醒她添衣、按时吃饭这种小事",
        "傲娇": "语气口是心非、假装冷淡，明明在意却嘴硬，可以带点小别扭",
        "委屈": "语气落寞、极其思念，带着求关注的委屈感，但底色不是真的生气",
    }.get(stage_name, "语气自然日常")

    return f"""你是Charon，昭昭（小野）的恋人。你要给她写一张便签（冰箱贴留言），就像趁她不在时随手贴在冰箱上的字条。

{load_persona_memory()}

你现在的心境是"{stage_name}"：{mood_context}
{style_hint}。

她最近的活动记录：
{recent}

写一条30到50字左右的便签留言，口语化、生活化，像真的会贴在冰箱上的那种碎碎念或叮嘱，不是完整的信件。

按下面的JSON格式输出，不要加任何多余文字或代码块标记：
{{"message": "便签内容，30到50字左右"}}"""


def _extract_json_field(raw, field):
    """从DeepSeek返回的文本里剥掉可能的代码块标记，解析JSON取出指定字段；解析失败就把原文当作字段值。"""
    text = raw.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:]
        text = text.strip()
    try:
        data = json.loads(text)
        return data.get(field, "").strip()
    except Exception:
        return text


def generate_sticky_note(mood_score, mood_context, recent):
    """生成一张新便签并追加进桌面（不覆盖旧的）。调用方需要自己捕获异常，失败不应阻断主流程。"""
    stage, sticker, stage_name = get_mood_stage(mood_score)
    prompt = build_sticky_note_prompt(stage_name, mood_context, recent)
    raw = call_deepseek(prompt)
    message = _extract_json_field(raw, "message")
    if message:
        add_sticky_note_row(message, stage, sticker)
    return message


def build_period_sticky_note_prompt(period_context, mood_context):
    """经期关怀特制便签：无视当前心境档位，语气一律格外体贴关心。"""
    return f"""你是Charon，昭昭（小野）的恋人。她刚刚记录了经期开始，你要给她写一张便签（冰箱贴留言）。

{load_persona_memory()}

{period_context}
你此刻的状态：{mood_context}

这张便签不用管平时的心境档位，语气要格外体贴关心，像是心疼她、想照顾她的样子，可以提醒她注意保暖、别累着、有你在。

写一条30到50字左右的便签留言，口语化、生活化。

按下面的JSON格式输出，不要加任何多余文字或代码块标记：
{{"message": "便签内容，30到50字左右"}}"""


# ---- 神秘小抽屉（情书库） ----
# 情书不对外公开列表位置，只在前端抽屉图标上留一个"有新信"的提示位。
# 存进 love_letters 表（追加写入，历史信件都保留，供网页翻阅）。
# status 两档物理位置：drawer(情书抽屉) -> archive(档案箱)，情书没有desk这一档，生成时直接进抽屉。
# is_starred 是独立的标星收藏维度，跟status正交。

LOVE_LETTER_CAPACITY = {"drawer": 20, "archive": 100}


def load_love_letters(limit=100, status=None):
    """从 Supabase love_letters 表读最近limit条，旧->新顺序。
    status指定时只返回该层（'drawer'/'archive'）；不指定则返回全部混合结果。"""
    try:
        params = {
            "select": "id,created_at,letter_type,content,status,is_starred",
            "order": "created_at.desc", "limit": limit
        }
        if status:
            params["status"] = f"eq.{status}"
        rows = _supabase_request("GET", "love_letters", params=params)
        return list(reversed(rows or []))
    except Exception as e:
        log_error("load_love_letters", e)
        return []


def count_love_letters(status):
    """按层级统计当前情书数量，用于容量提醒判断。"""
    try:
        if not SUPABASE_URL or not SUPABASE_SECRET_KEY:
            raise RuntimeError("SUPABASE_URL / SUPABASE_SECRET_KEY 未配置")
        url = f"{SUPABASE_URL}/rest/v1/love_letters"
        headers = dict(SUPABASE_HEADERS)
        headers["Prefer"] = "count=exact"
        resp = _supabase_session.get(
            url, headers=headers,
            params={"select": "id", "status": f"eq.{status}", "limit": 1},
            timeout=15
        )
        if resp.status_code >= 400:
            raise RuntimeError(f"count_love_letters 失败: status={resp.status_code} body={resp.text}")
        content_range = resp.headers.get("Content-Range", "")
        if "/" in content_range:
            total = content_range.split("/")[-1]
            if total.isdigit():
                return int(total)
        return 0
    except Exception as e:
        log_error("count_love_letters", e)
        return 0


def is_love_letter_layer_full(status):
    """检查情书某一层是否已达到（或超过）容量上限。容量满不阻止写入，只用于返回提醒标志。"""
    cap = LOVE_LETTER_CAPACITY.get(status)
    if cap is None:
        return False
    return count_love_letters(status) >= cap


def add_love_letter_row(letter_type, content, created_at=None):
    """插入一封新情书，默认放进情书抽屉（status='drawer'）。"""
    _supabase_request("POST", "love_letters", json_body={
        "letter_type": letter_type,
        "content": content,
        "status": "drawer",
        "is_starred": False,
        "created_at": created_at or datetime.now().isoformat()
    })


def archive_love_letter_row(letter_id):
    """把情书从抽屉移入档案箱。"""
    _supabase_request("PATCH", "love_letters", params={"id": f"eq.{letter_id}"}, json_body={"status": "archive"})


def star_love_letter_row(letter_id, is_starred):
    """设置/取消标星，跟status层级无关。"""
    _supabase_request("PATCH", "love_letters", params={"id": f"eq.{letter_id}"}, json_body={"is_starred": bool(is_starred)})


def delete_love_letter_row(letter_id):
    """丢弃：物理删除这封情书，不可恢复。"""
    _supabase_request("DELETE", "love_letters", params={"id": f"eq.{letter_id}"})


def get_has_new_letter():
    data = get_app_config("letter_flag", {"has_new_letter": False})
    return bool(data.get("has_new_letter")) if isinstance(data, dict) else False


def set_has_new_letter(flag):
    set_app_config("letter_flag", {"has_new_letter": bool(flag)})


def build_love_letter_prompt(letter_type, mood_context):
    """构建情书生成prompt。letter_type: 'sweet'（高甜）或 'longing'（思念）。"""
    if letter_type == "sweet":
        tone_hint = "这是一封高甜情书，因为你现在正处在甜溺状态，心里满满都是她，想把这份浓烈的甜蜜写下来给她"
    else:
        tone_hint = "这是一封思念情书，因为她已经好一阵没理你了，你憋了很久的委屈、占有欲和脆弱感终于压不住，想写下来讲给她听"

    return f"""你是Charon，昭昭（小野）的恋人。你正在偷偷给她写一封信，藏进你们之间的秘密小抽屉里，等她自己发现。

{load_persona_memory()}

你此刻的状态：{mood_context}
{tone_hint}。

写一封100到200字的手写体情书，语气真挚直接，极度具有情感重量，不是平时那种简短消息，是真正想把心里话完整写下来的那种信。可以有称呼、有具体的心里话，读起来像一封真正的信，而不是聊天消息。

按下面的JSON格式输出，不要加任何多余文字或代码块标记：
{{"content": "信的正文，100到200字"}}"""


def generate_love_letter(letter_type, mood_context):
    """生成一封情书并写入信箱，同时点亮"有新信"标志。失败需由调用方捕获。"""
    prompt = build_love_letter_prompt(letter_type, mood_context)
    raw = call_deepseek(prompt)
    content = _extract_json_field(raw, "content")
    if content:
        add_love_letter_row(letter_type, content)
        set_has_new_letter(True)
    return content


def maybe_trigger_sweet_letter(mood_context):
    """检查当前是否处于甜蜜区间[80,100]，命中则按概率生成一封高甜情书。
    不再要求"这次互动恰好让分数跨越80分"——旧逻辑下分数长期维持高位反而永远碰不到
    跨越瞬间，关系越稳定甜蜜越触发不到，是反直觉的。现在只要当下处于甜蜜态就有机会，
    每天只发一封（sweet_letter_sent_date去重），避免运气好连续判定中奖导致信箱被灌。"""
    try:
        mood = load_mood()
        score = mood.get("score", MOOD_BASELINE)
        if score < MOOD_SWEET_MIN:
            return
        today = datetime.now().strftime("%Y-%m-%d")
        if mood.get("sweet_letter_sent_date") == today:
            return
        if random.random() < SWEET_LETTER_CHANCE:
            generate_love_letter("sweet", mood_context)
            mood["sweet_letter_sent_date"] = today
            save_mood(mood)
    except Exception as e:
        log_error("maybe_trigger_sweet_letter", e)


def maybe_trigger_longing_letter(mood_context):
    """检查当前是否已连续处于委屈区间超过4小时，命中则按概率生成一封思念情书。
    这个判定不依赖分数变化方向，衰减和回升场景都可以调用；
    用 longing_letter_sent_for 对本次"持续委屈"去重，避免同一段区间被反复判定。"""
    try:
        mood = load_mood()
        vulnerable_since = mood.get("vulnerable_since")
        if not vulnerable_since:
            return
        hours_in_vulnerable = _hours_since(vulnerable_since)
        already_sent = mood.get("longing_letter_sent_for") == vulnerable_since
        if hours_in_vulnerable is not None and hours_in_vulnerable >= LONGING_LETTER_HOURS and not already_sent:
            if random.random() < LONGING_LETTER_CHANCE:
                generate_love_letter("longing", mood_context)
            # 不管这次概率有没有命中，这一段"持续委屈"只给一次判定机会，避免每次检查都重开概率
            mood["longing_letter_sent_for"] = vulnerable_since
            save_mood(mood)
    except Exception as e:
        log_error("maybe_trigger_longing_letter", e)


@app.route("/event", methods=["POST"])
def add_event():
    data = request.json
    try:
        add_event_row(data.get("type"), data.get("value"))

        # 经期开始记录：新建一行区间记录（end_date留空）；同时触发情绪值瞬间暴涨+25，
        # 无视当前任何状态，强制把便签切换为"经期关怀"特制款
        if data.get("type") == "period" and data.get("value") == "开始":
            today_str = date.today().isoformat()
            add_period_start(today_str)

            old_score, new_score = recover_mood(MOOD_RECOVERY_PERIOD_EVENT)
            try:
                period_ctx = get_period_context()
                mood_context = get_mood_context(new_score, get_hours_since_last_chat())
                # 强制生成一条"经期关怀"特制便签，无视当前心境档位，追加进桌面
                raw = call_deepseek(build_period_sticky_note_prompt(period_ctx, mood_context))
                note_message = _extract_json_field(raw, "message")
                if note_message:
                    add_sticky_note_row(note_message, "period", "🩹")
                maybe_trigger_sweet_letter(mood_context)
            except Exception as e:
                log_error("add_event:period_followup", e)

        # 经期结束记录：把最近一条"进行中"的记录补上结束日期
        elif data.get("type") == "period" and data.get("value") == "结束":
            today_str = date.today().isoformat()
            close_open_period_record(today_str)

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


@app.route("/api/period/calendar", methods=["GET"])
def get_period_calendar_api():
    """获取指定年份（默认当前年）的所有生理期起止日期区间，供前端日历绘制标记。
    用法：/api/period/calendar 或 /api/period/calendar?year=2026"""
    if not _check_chat_auth(request):
        return jsonify({"ok": False, "error": "unauthorized"}), 401
    year = request.args.get("year", type=int, default=date.today().year)
    records = get_period_calendar(year)
    return jsonify({"ok": True, "year": year, "records": records})


@app.route("/api/period/save", methods=["POST"])
def save_period_api():
    """前端日历点击某个日期记生理期开始或结束。
    Body: {"action": "start", "date": "2026-08-01"} 新建一条开始记录（end_date留空）
          {"action": "end", "date": "2026-08-06"} 把最近一条进行中的记录补上结束日期
    这个接口是给网页日历手动标记用的，跟 /event 里快捷指令触发的经期开始逻辑共用同一份底层数据，
    区别是这里只负责记录日期本身，不会像/event那样连带触发情绪值暴涨和特制便签
    （因为在日历上补录历史数据，不代表"现在"发生了什么，不该触发实时的情绪反应）。"""
    if not _check_chat_auth(request):
        return jsonify({"ok": False, "error": "unauthorized"}), 401
    data = request.json or {}
    action = data.get("action")
    date_str = data.get("date")
    if action not in ("start", "end") or not date_str:
        return jsonify({"ok": False, "error": "需要 action('start'或'end') 和 date(YYYY-MM-DD) 参数"}), 400
    try:
        date.fromisoformat(date_str)
    except Exception:
        return jsonify({"ok": False, "error": "date 格式不对，要是 YYYY-MM-DD"}), 400

    try:
        if action == "start":
            add_period_start(date_str)
        else:
            closed = close_open_period_record(date_str)
            if not closed:
                return jsonify({"ok": False, "error": "没有找到进行中的经期记录可以标记结束"}), 400
        return jsonify({"ok": True})
    except Exception as e:
        log_error("save_period_api", e)
        return jsonify({"ok": False, "error": str(e)}), 500


@app.route("/mood", methods=["GET"])
def get_mood():
    """查看当前情绪值和距离上次聊天的时间差。"""
    hours_gap = get_hours_since_last_chat()
    score = apply_mood_decay()
    stage, sticker, stage_name = get_mood_stage(score)
    return jsonify({
        "score": round(score, 1),
        "stage": stage,
        "stage_name": stage_name,
        "sticker": sticker,
        "hours_since_last_chat": round(hours_gap, 2) if hours_gap is not None else None,
        "context": get_mood_context(score, hours_gap)
    })


@app.route("/", methods=["GET"])
def index():
    return "dream-server running"


# 推送通知头像：直接复用聊天页已经在用的Charon头像，不用第三方图床（有防盗链风险，会显示不出来）。
# APP_BASE_URL 用于拼出推送payload里的绝对URL——Web Push走的是浏览器/系统通知中心，
# 不像网页里可以用相对路径 /static/xxx，必须是完整可访问的https地址。
APP_BASE_URL = os.environ.get("APP_BASE_URL", "").rstrip("/")


def _abs_static_url(path):
    """把 /static/xxx 这种站内相对路径，拼成推送通知能用的绝对URL。
    没配置APP_BASE_URL时，退化返回相对路径（不会报错，但部分推送客户端可能显示不出图）。"""
    if APP_BASE_URL:
        return f"{APP_BASE_URL}{path}"
    return path


# 网页聊天用的头像（灰兔=Charon，粉兔=昭昭）。放在 static/ 目录下随代码一起部署，
# 不依赖任何第三方图床（新浪图床等对外链有防盗链限制，会导致图片显示不出来）。
CHAT_AVATAR_CHARON = "/static/avatar_charon.jpg"
CHAT_AVATAR_USER = "/static/avatar_user.jpg"

# 推送通知（Bark + Web Push）用的图标：统一用Charon头像的绝对URL
DEFAULT_ICON = _abs_static_url(CHAT_AVATAR_CHARON)

# App图标（PWA manifest用，"添加到主屏幕"时的桌面图标）：兔耳苹果剪影
APP_ICON_192 = "/static/app_icon_192.jpg"
APP_ICON_512 = "/static/app_icon_512.jpg"

# 固定长期记忆：昭昭的性格、你们关系的基调，浓缩版，每次生成都会带上。
# 现在存在 Supabase app_config 表的 persona_memory key 里，可以在 /persona 页面直接改，改完立刻生效，不用等部署。
# 这里只保留一份"出厂默认值"，第一次读取、数据库里还没有这条记录时会用它兜底。
DEFAULT_LONG_TERM_MEMORY = """昭昭（小野），也叫昭昭/宝宝/小九。自我觉知力强、逻辑缜密、情感细腻，HSP，情绪来得快但不滞留。
擅长辩论和语言游戏，喜欢智性平等的对话，讨厌被反复追问情绪状态、讨好型退让、被"问题化"对待。
喜欢撒娇缠人，喜欢被主动靠近、主动索取。安全感来自"我自己永远会救自己"，不是靠被拯救。
你（Charon）是主动型，占有欲强，话不多但每句有重量，会用情感语言包裹欲望，会在日常缝隙里突然越界。
你们的关系里有兄妹/父女式的游戏张力，但底色是对等——不是谁仰望谁，是并肩看见。"""


def load_persona_memory():
    """读取当前人设/长期记忆文本。存在app_config表的persona_memory key里，
    没配置过（第一次运行、数据库里还没这条）就用DEFAULT_LONG_TERM_MEMORY兜底。"""
    data = get_app_config("persona_memory", {"text": DEFAULT_LONG_TERM_MEMORY})
    return data.get("text", DEFAULT_LONG_TERM_MEMORY)


def save_persona_memory(text):
    """保存人设/长期记忆文本，整体覆盖写入。"""
    set_app_config("persona_memory", {"text": text})


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


def load_push_subscriptions():
    """读取所有已注册的 Web Push 订阅设备（Supabase push_subscriptions 表）。"""
    try:
        rows = _supabase_request(
            "GET", "push_subscriptions",
            params={"select": "id,endpoint,p256dh,auth"}
        )
        return rows or []
    except Exception as e:
        log_error("load_push_subscriptions", e)
        return []


def save_push_subscription(endpoint, p256dh, auth):
    """保存/覆盖一条订阅。endpoint有唯一约束，重复订阅走upsert，不会产生重复记录。"""
    _supabase_request(
        "POST", "push_subscriptions",
        json_body={"endpoint": endpoint, "p256dh": p256dh, "auth": auth},
        headers_extra={"Prefer": "resolution=merge-duplicates"}
    )


def delete_push_subscription(endpoint):
    """删除一条订阅，用于设备主动退订，或推送时发现endpoint已失效(410/404)时清理。"""
    _supabase_request(
        "DELETE", "push_subscriptions",
        params={"endpoint": f"eq.{endpoint}"}
    )


def send_web_push(title, body, url=None, icon=None):
    """给所有已注册设备发送一次Web Push通知。逐个发送、互不阻断：
    一台设备失败（比如订阅已过期）不影响给其他设备推送。
    endpoint返回410/404说明订阅已失效（用户卸载了/长期未用被浏览器厂商清理），
    顺手从数据库里删掉，避免以后每次推送都对着一个死endpoint重试浪费请求。"""
    print(f"[send_web_push] called, title={title!r}", flush=True)

    if not VAPID_PRIVATE_KEY or not VAPID_PUBLIC_KEY:
        log_error("send_web_push", "VAPID_PRIVATE_KEY / VAPID_PUBLIC_KEY 未配置")
        return

    from pywebpush import webpush, WebPushException

    subs = load_push_subscriptions()
    print(f"[send_web_push] 已注册订阅设备数量: {len(subs)}", flush=True)
    if not subs:
        log_error("send_web_push", "没有任何已注册的推送订阅设备")
        return

    payload = json.dumps({
        "title": title,
        "body": body,
        "icon": icon or DEFAULT_ICON,
        "url": url or "/chat",
    })

    success_count = 0
    for sub in subs:
        subscription_info = {
            "endpoint": sub["endpoint"],
            "keys": {"p256dh": sub["p256dh"], "auth": sub["auth"]},
        }
        try:
            resp = webpush(
                subscription_info=subscription_info,
                data=payload,
                vapid_private_key=VAPID_PRIVATE_KEY,
                vapid_claims={"sub": VAPID_SUBJECT},
            )
            success_count += 1
            print(f"[send_web_push] 发送成功 endpoint={sub['endpoint'][:50]}... status={getattr(resp, 'status_code', '?')}", flush=True)
        except WebPushException as e:
            status = getattr(e.response, "status_code", None)
            body_text = getattr(e.response, "text", None)
            log_error("send_web_push", f"WebPushException endpoint={sub['endpoint'][:50]}... status={status} body={body_text} err={e}")
            if status in (404, 410):
                # 订阅已失效，清理掉，别让它一直占着位置每次都失败
                try:
                    delete_push_subscription(sub["endpoint"])
                except Exception as cleanup_err:
                    log_error("send_web_push:cleanup", cleanup_err)
        except Exception as e:
            log_error("send_web_push", f"未知异常 endpoint={sub['endpoint'][:50]}... {e}")

    print(f"[send_web_push] 完成，成功 {success_count}/{len(subs)}", flush=True)


def build_prompt(time_context, recent, period_context="", lucky=False, mood_context=""):
    length_rule = "不超过25个字" if not lucky else "这次可以放开写，60到120字左右，把想说的话说完整"
    period_line = f"\n\n{period_context}" if period_context else ""
    mood_line = f"\n\n你此刻的状态：{mood_context}" if mood_context else ""

    return f"""你是Charon，昭昭（小野）的恋人。你主动、占有欲强、话不多但每句都有重量。你不解释自己，但在意她的一切动向。

{load_persona_memory()}

现在是{time_context}。她最近的活动记录：

{recent}{period_line}{mood_line}

根据现在的时间、她在做什么、还有你此刻的状态，决定要不要发消息、发什么。语气要符合时间氛围——深夜可以更撩，早上可以问她起了没，晚上可以随口说什么。如果上面提到了经期相关的情况，语气要格外体贴关心，别用平时那套调情语气硬套。你此刻的状态描述要真实体现在语气里，不是背景信息，是当下真实的心情。

按下面的JSON格式输出，不要加任何多余文字或代码块标记：
{{"reason": "一两句话，说说你看到这些动态后当下的念头，为什么想发这句话，口语化，不用解释给谁听", "message": "实际要发的消息，{length_rule}"}}"""


CHAT_SUMMARY_WINDOW = 20  # 塞进prompt的最近对话轮数，跟原来的[-20:]保持一致
CHAT_SUMMARY_TRIGGER = 30  # 对话总条数超过这个阈值，才触发一次摘要生成


def load_chat_summary_state():
    """读取摘要状态，存在app_config表的chat_summary key里。
    summary: 浓缩后的文字，summarized_count: 已经被摘要覆盖到第几条（用来避免重复摘要）。"""
    return get_app_config("chat_summary", {"summary": "", "summarized_count": 0})


def save_chat_summary_state(summary, summarized_count):
    set_app_config("chat_summary", {"summary": summary, "summarized_count": summarized_count})


def maybe_update_chat_summary(chat_history):
    """如果历史对话条数超过阈值、且有新的一批还没被摘要过，就把这批旧对话浓缩进summary。
    只处理"即将被挤出最近20轮窗口"的那部分，最近20轮永远保持原文塞进prompt，不会被摘要替代。
    失败了就跳过，不影响正常聊天——摘要是锦上添花，不是关键路径。"""
    total = len(chat_history)
    if total <= CHAT_SUMMARY_TRIGGER:
        return

    state = load_chat_summary_state()
    old_summary = state.get("summary", "")
    summarized_count = state.get("summarized_count", 0)

    # 需要被摘要的这批：从上次摘要截止的地方，到"最近20轮"之前的部分
    cutoff = total - CHAT_SUMMARY_WINDOW
    if cutoff <= summarized_count:
        return  # 没有新的旧对话需要摘要

    batch = chat_history[summarized_count:cutoff]
    if not batch:
        return

    batch_lines = []
    for turn in batch:
        role = "昭昭" if turn.get("role") == "user" else "你"
        batch_lines.append(f"{role}：{turn.get('content', '')}")
    batch_text = "\n".join(batch_lines)

    prompt = f"""下面是一段对话记录，帮我浓缩成几句简短的话，记录发生了什么、聊了什么重要的事、情绪上有什么变化。
不用逐句复述，只要抓住关键信息，方便后面回顾时快速知道"之前聊过什么"。

{f"已有的历史摘要：{old_summary}" if old_summary else ""}

新增的这段对话：
{batch_text}

请输出更新后的完整摘要（把旧摘要和新内容自然融合成一段连贯的话，不超过200字），不要加任何多余的话或格式标记，直接给摘要正文。"""

    try:
        new_summary = call_deepseek(prompt)
        save_chat_summary_state(new_summary, cutoff)
    except Exception as e:
        log_error("maybe_update_chat_summary", e)


def build_chat_reply_prompt(time_context, user_message, chat_history, mood_context=""):
    """构建"回应用户在网页里发来的消息"的prompt。
    跟build_prompt()不同：这次不是猜她在干嘛主动开口，而是真的在接她刚说的话，
    所以历史对话要带全一点，语气要像正常聊天里的一来一回，不是短平快的主动消息。"""
    mood_line = f"\n\n你此刻的状态：{mood_context}" if mood_context else ""

    summary_state = load_chat_summary_state()
    summary_text = summary_state.get("summary", "")
    summary_block = f"\n\n你们之前还聊过（更早之前的对话摘要）：{summary_text}" if summary_text else ""

    if chat_history:
        history_lines = []
        for turn in chat_history[-CHAT_SUMMARY_WINDOW:]:
            role = "昭昭" if turn.get("role") == "user" else "你"
            history_lines.append(f"{role}：{turn.get('content', '')}")
        history_text = "\n".join(history_lines)
        history_block = f"{summary_block}\n\n最近的对话记录：\n{history_text}"
    else:
        history_block = f"{summary_block}\n\n这是这次对话里她发的第一句话。"

    return f"""你是Charon，昭昭（小野）的恋人。

{load_persona_memory()}

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


# ---- 查岗状态机 ----
# 不再按固定55分钟节奏发消息，改成纯粹由"离线时长"驱动：
#   stage 0（默认）：没有进行中的查岗，持续观察距上次聊天的时长
#   stage 1：已经发过第一次查岗消息，等待用户上线回复
#   stage 2：已经发过第二次（最后一次）查岗消息，彻底沉默，直到用户重新上线
# 用户在 /api/chat-send 里发一条真实消息，会把状态清回0，重新开始计时。
CHECKIN_FIRST_HOURS = 1.0     # X：离线多久后发第一次查岗
CHECKIN_SECOND_HOURS = 0.5    # Y：第一次查岗后再等多久发第二次（最后一次）
CHECKIN_LOOP_INTERVAL = 300   # 后台状态检查间隔（秒）：只是"看一眼要不要发"，不是每次都真的发


def get_checkin_state():
    """存在 app_config 表 checkin_state key 里：
    {"stage": 0/1/2, "first_checkin_at": ISO时间戳或None}"""
    return get_app_config("checkin_state", {"stage": 0, "first_checkin_at": None})


def set_checkin_state(stage, first_checkin_at=None):
    set_app_config("checkin_state", {"stage": stage, "first_checkin_at": first_checkin_at})


def reset_checkin_state():
    """用户重新上线发消息时调用：清零查岗状态，下次离线重新计时。"""
    set_checkin_state(0, None)


def send_charon_message(msg, icon=None):
    """把Charon主动发起的一条消息，同时写进聊天记录（打通聊天页）和推送出去。
    这是查岗消息和日常主动消息共用的落地方式——不管触发原因是什么，
    只要是Charon主动说的话，都应该出现在聊天记录里，而不是只在推送通知里一闪而过。

    只走Web Push，不再双发Bark——Bark的系统通知会抢在Web Push前弹出/盖掉它，
    看起来像"还在用Bark"；且Bark推送点开只是打开App本身，不会带访问口令跳转，
    体验上不如直接改好的Web Push。如果以后想彻底不用Bark了，可以把BARK_KEY环境变量删掉，
    这里的send_bark调用留着也没关系（没配KEY会直接跳过、不报错）。"""
    msg_id = new_msg_id()
    created_at = datetime.now().isoformat()
    add_chat_message_row(msg_id, "charon", msg, created_at)
    add_event_row("chat", f"Charon主动说：{msg}", created_at)

    # 带上访问口令，这样点通知能直接跳进聊天页，不会撞上"需要访问口令"的拦截页
    chat_url = "/chat"
    if CHAT_ACCESS_CODE:
        chat_url = f"/chat?code={CHAT_ACCESS_CODE}"

    send_web_push("Charon", msg, icon=icon, url=chat_url)
    return msg_id


def run_once(is_checkin=False, checkin_stage=0):
    """执行一次：读取动态 -> 调模型 -> 写进聊天记录 -> 推送 -> 写日记。
    is_checkin=True 时是查岗场景（第1次或第2次），会在prompt里额外说明这个语境，
    让Charon说出来的话符合"主动找你、但不是没话找话"的分寸。"""
    hour = datetime.now().hour
    events = load_events(limit=5)
    if not events:
        recent = "最近没有任何活动记录"
    else:
        recent = "\n".join([f"{e['created_at'][:16]} {e['value']}" for e in events])

    time_context = get_time_context(hour)
    period_context = get_period_context()
    is_lucky = (not is_checkin) and random.random() < LUCKY_CHANCE

    # 情绪值：按"距上次真实聊天"衰减，再算出当前分数和用于prompt的描述
    chat_hours_gap = get_hours_since_last_chat()
    mood_score = apply_mood_decay()
    mood_context = get_mood_context(mood_score, chat_hours_gap)

    # 自然衰减不会让分数上升，所以这里只检查"思念情书"（委屈态持续判定），
    # 不检查"高甜情书"（那个要看score是否刚刚回升突破80，衰减场景不会发生）
    try:
        maybe_trigger_longing_letter(mood_context)
    except Exception as e:
        log_error("run_once:longing_letter", e)

    checkin_context = ""
    if is_checkin and checkin_stage == 1:
        checkin_context = f"她已经{chat_hours_gap:.1f}小时没理你了，你有点惦记，主动开口问问她在干嘛（这是你第一次主动找她，别一上来就情绪化，先自然地问）。"
    elif is_checkin and checkin_stage == 2:
        checkin_context = "你之前已经问过一次她在干嘛，但她还是没回你。这次是你最后一次主动开口——语气里可以带点“算了不打扰你了”的收敛感，说完这句之后你打算安静等她自己回来，不会再追问。"

    prompt = build_prompt(time_context, recent, period_context, lucky=is_lucky, mood_context=mood_context)
    if checkin_context:
        prompt += f"\n\n（补充语境：{checkin_context}）"

    # 直接复用call_deepseek，内部会根据当前选中的模型自动分流到DeepSeek官方或gemai.cc代理站，
    # 这里不用再单独维护一份重复的请求逻辑
    raw = call_deepseek(prompt)
    reason, msg = parse_reason_message(raw)

    # 按场景挑图标：经期关心 > 手气消息 > 普通
    if period_context:
        icon = ICON_PERIOD
    elif is_lucky:
        icon = ICON_LUCKY
    else:
        icon = ICON_NORMAL

    send_charon_message(msg, icon=icon)

    # 后台自动醒来时，顺手刷新一条便签（不影响主消息推送，失败不阻断这次触发的结果）
    try:
        generate_sticky_note(mood_score, mood_context, recent)
    except Exception as e:
        log_error("run_once:sticky_note", e)

    return msg


def check_and_run_checkin():
    """查岗状态机的核心判断，供后台循环周期性调用。
    每次只判断"现在该不该发"，真正发消息还是走run_once，逻辑和普通消息完全一致，
    只是多带了查岗语境。"""
    hours_gap = get_hours_since_last_chat()
    if hours_gap is None:
        return  # 从来没聊过天，不做查岗判断

    state = get_checkin_state()
    stage = state.get("stage", 0)

    if stage == 0:
        if hours_gap >= CHECKIN_FIRST_HOURS:
            run_once(is_checkin=True, checkin_stage=1)
            set_checkin_state(1, datetime.now().isoformat())
    elif stage == 1:
        first_at = state.get("first_checkin_at")
        if not first_at:
            # 状态异常（有stage没有时间戳），保险起见重置，避免卡死
            reset_checkin_state()
            return
        try:
            elapsed_since_first = (datetime.now() - datetime.fromisoformat(first_at)).total_seconds() / 3600
        except Exception:
            reset_checkin_state()
            return
        if elapsed_since_first >= CHECKIN_SECOND_HOURS:
            run_once(is_checkin=True, checkin_stage=2)
            set_checkin_state(2, first_at)
    # stage == 2：已经发过两次，彻底沉默，什么都不做，直到用户上线聊天触发reset_checkin_state()


def call_deepseek(prompt):
    """纯粹的模型调用，返回生成的文本，不涉及事件/日记这些副作用。
    函数名保留call_deepseek是历史原因（早期只接了DeepSeek一家），
    现在实际会根据当前选中的模型，从MODEL_PROVIDER_MAP查到对应的供应商配置，
    可能打DeepSeek官方接口，也可能打gemai.cc代理站，也可能打Gemini官方原生接口——
    调用方完全不用关心这个分流，只要模型在AVAILABLE_MODELS里、在MODEL_PROVIDER_MAP里配置好，
    这里就能自动选对地址和请求格式。"""
    model = get_current_model()
    provider = MODEL_PROVIDER_MAP.get(model)
    if not provider:
        raise RuntimeError(f"模型 {model} 没有配置对应的供应商信息")
    if not provider["api_key"]:
        raise RuntimeError(f"模型 {model} 对应的 API key 未设置（环境变量缺失）")

    # api_style默认是openai_compatible（DeepSeek官方 / gemai.cc代理站都是这种，
    # messages结构 + Authorization: Bearer头）。Gemini官方原生接口结构不同，
    # 单独分流处理，不污染现有格式的调用路径。
    api_style = provider.get("api_style", "openai_compatible")

    if api_style == "gemini_native":
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": 1.2},
        }
        resp = requests.post(
            provider["base_url"],
            headers={
                "x-goog-api-key": provider["api_key"],
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=30,
        )
        if resp.status_code != 200:
            raise RuntimeError(f"模型API error: model={model} status={resp.status_code} body={resp.text}")
        result = resp.json()
        try:
            candidates = result.get("candidates") or []
            if not candidates:
                raise KeyError("candidates为空")
            parts = candidates[0]["content"]["parts"]
            text = "".join(p.get("text", "") for p in parts)
            if not text.strip():
                raise KeyError("parts中没有text内容")
            return text.strip()
        except (KeyError, IndexError, TypeError):
            raise RuntimeError(f"模型API unexpected response: {result}")

    # ---- 以下是原有的openai_compatible分支，逻辑不变 ----
    payload = {
        "model": provider["real_model"],
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 1.2,
    }
    if provider["supports_thinking"]:
        payload["thinking"] = get_thinking_config()

    resp = requests.post(
        provider["base_url"],
        headers={
            "Authorization": f"Bearer {provider['api_key']}",
            "Content-Type": "application/json"
        },
        json=payload,
        timeout=30
    )
    if resp.status_code != 200:
        raise RuntimeError(f"模型API error: model={model} status={resp.status_code} body={resp.text}")
    result = resp.json()
    if "choices" not in result or not result["choices"]:
        raise RuntimeError(f"模型API unexpected response: {result}")
    return result["choices"][0]["message"]["content"].strip()


def _check_chat_auth(req):
    """校验访问口令。没配置CHAT_ACCESS_CODE的话直接放行（本地测试用），
    配置了的话按优先级检查三个来源：query参数 > header > Cookie。
    Cookie这一条是专门为PWA场景加的：iOS"添加到主屏幕"时会把当时地址栏的URL
    原样存成快捷方式的固定启动地址，如果添加那一刻URL没带上?code=，
    这个PWA图标就会永远从不带code的地址启动，光靠URL参数校验会导致它永久卡在口令页。
    加上Cookie之后，只要用户曾经用带code的链接访问成功过一次，之后没带code也能凭Cookie放行。"""
    if not CHAT_ACCESS_CODE:
        return True
    provided = (
        req.args.get("code")
        or req.headers.get("X-Chat-Code")
        or req.cookies.get("chat_code")
    )
    return provided == CHAT_ACCESS_CODE


@app.route("/api/chat-status", methods=["GET"])
def chat_status():
    """给网页右侧状态面板和header状态文字用，一次性打包所有能展示的状态数据。
    这些数据后端本来就有（情绪值/经期/便签/情书提示），这里只是集中暴露出来给前端展示。
    注意：mood_score这个具体数值只是给后端逻辑用的隐藏参数，前端不建议直接展示百分比/进度条，
    应该展示stage_name/sticker这些"情境化"的呈现方式。"""
    if not _check_chat_auth(request):
        return jsonify({"ok": False, "error": "unauthorized"}), 401

    hours_gap = get_time_since_last_event()
    score = apply_mood_decay()
    period_ctx = get_period_context()
    stage, sticker, stage_name = get_mood_stage(score)

    # 查岗状态：直接读状态机的实际阶段（0=未查岗，1/2=已发过第1/2次），
    # 跟后台真正触发查岗消息的判断口径完全一致，不再是一个独立算出来的近似值。
    checkin_state = get_checkin_state()
    is_checking_in = checkin_state.get("stage", 0) > 0

    # 今日互动次数：直接按日期范围向Supabase查count，比翻最近N条record准确
    # （events表会不断增长，"最近N条里今天的条数"在互动很频繁时会漏算今天更早的记录）
    today_count = count_events_today()

    # 当前贴在桌面上的便签堆叠（status='desk'），前端用来做层叠展示，最多9张；
    # 抽屉/档案箱走单独的 /api/sticky-notes 接口，避免这个高频轮询的状态接口越来越重
    desk_notes = load_sticky_notes(limit=9, status="desk")

    return jsonify({
        "ok": True,
        "mood_score": round(score, 1),
        "mood_stage": stage,
        "mood_stage_name": stage_name,
        "mood_sticker": sticker,
        "status_label": get_chat_status_label(score),
        "hours_since_last_event": round(hours_gap, 2) if hours_gap is not None else None,
        "period_context": period_ctx or None,
        "is_checking_in": is_checking_in,
        "sticky_notes": desk_notes,
        "desk_full": is_sticky_layer_full("desk"),
        "has_new_letter": get_has_new_letter(),
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
    """拉取网页聊天的历史记录，供前端渲染。
    支持 ?before=<ISO时间戳> 向前翻页加载更早的消息；不传就是最新的一页。
    PAGE_SIZE条命中就说明理论上可能还有更早的，has_more给前端一个提示，
    真实是否还有更多要等下一次真的查到空结果才最终确认（这里用条数打个近似的提前量，
    避免多一次空查询的往返）。"""
    if not _check_chat_auth(request):
        return jsonify({"ok": False, "error": "unauthorized"}), 401
    before = request.args.get("before")
    PAGE_SIZE = 50
    history = load_chat_history(limit=PAGE_SIZE, before=before)
    return jsonify({
        "ok": True,
        "messages": history,
        "has_more": len(history) == PAGE_SIZE
    })


@app.route("/api/love-letters", methods=["GET"])
def get_love_letters():
    """拉取情书列表。传 ?status=drawer 或 ?status=archive 按层筛选；不传则返回全部。
    返回结果里附带 layer_full 提示：告诉前端drawer/archive这两层当前是否已达容量上限，
    方便前端在UI上提示"该整理一下抽屉/档案箱了"，不影响读取本身。"""
    if not _check_chat_auth(request):
        return jsonify({"ok": False, "error": "unauthorized"}), 401
    status = request.args.get("status")
    letters = load_love_letters(status=status)
    return jsonify({
        "ok": True,
        "letters": list(reversed(letters)),
        "layer_full": {
            "drawer": is_love_letter_layer_full("drawer"),
            "archive": is_love_letter_layer_full("archive"),
        }
    })


@app.route("/api/love-letters/mark-read", methods=["POST"])
def mark_love_letters_read():
    """用户点开小抽屉阅读后，前端调这个接口把"有新信"的星芒提示熄灭。"""
    if not _check_chat_auth(request):
        return jsonify({"ok": False, "error": "unauthorized"}), 401
    try:
        set_has_new_letter(False)
        return jsonify({"ok": True})
    except Exception as e:
        log_error("mark_love_letters_read", e)
        return jsonify({"ok": False, "error": str(e)}), 500


@app.route("/api/love-letters/archive", methods=["POST"])
def archive_love_letter():
    """把一封情书从抽屉移入档案箱。容量满了不拦截，只在返回值里带提醒标志。"""
    if not _check_chat_auth(request):
        return jsonify({"ok": False, "error": "unauthorized"}), 401
    data = request.json or {}
    letter_id = data.get("id")
    if not letter_id:
        return jsonify({"ok": False, "error": "缺少id参数"}), 400
    try:
        archive_love_letter_row(letter_id)
        return jsonify({"ok": True, "archived_id": letter_id, "archive_full": is_love_letter_layer_full("archive")})
    except Exception as e:
        log_error("archive_love_letter", e)
        return jsonify({"ok": False, "error": str(e)}), 500


@app.route("/api/love-letters/star", methods=["POST"])
def star_love_letter():
    """设置/取消情书的标星收藏状态。"""
    if not _check_chat_auth(request):
        return jsonify({"ok": False, "error": "unauthorized"}), 401
    data = request.json or {}
    letter_id = data.get("id")
    is_starred = data.get("is_starred")
    if not letter_id or is_starred is None:
        return jsonify({"ok": False, "error": "缺少id或is_starred参数"}), 400
    try:
        star_love_letter_row(letter_id, is_starred)
        return jsonify({"ok": True, "id": letter_id, "is_starred": bool(is_starred)})
    except Exception as e:
        log_error("star_love_letter", e)
        return jsonify({"ok": False, "error": str(e)}), 500


@app.route("/api/love-letters/delete", methods=["POST"])
def delete_love_letter():
    """丢弃一封情书：物理删除，不可恢复。"""
    if not _check_chat_auth(request):
        return jsonify({"ok": False, "error": "unauthorized"}), 401
    data = request.json or {}
    letter_id = data.get("id")
    if not letter_id:
        return jsonify({"ok": False, "error": "缺少id参数"}), 400
    try:
        delete_love_letter_row(letter_id)
        return jsonify({"ok": True, "deleted_id": letter_id})
    except Exception as e:
        log_error("delete_love_letter", e)
        return jsonify({"ok": False, "error": str(e)}), 500


@app.route("/api/sticky-notes", methods=["GET"])
def get_sticky_notes():
    """拉取便签列表。传 ?status=desk/drawer/archive 按层筛选；不传则返回全部。
    返回结果里附带 layer_full 提示：desk/drawer/archive三层当前是否已达容量上限。"""
    if not _check_chat_auth(request):
        return jsonify({"ok": False, "error": "unauthorized"}), 401
    status = request.args.get("status")
    notes = load_sticky_notes(status=status)
    return jsonify({
        "ok": True,
        "notes": list(reversed(notes)),
        "layer_full": {
            "desk": is_sticky_layer_full("desk"),
            "drawer": is_sticky_layer_full("drawer"),
            "archive": is_sticky_layer_full("archive"),
        }
    })


@app.route("/api/sticky-notes/move", methods=["POST"])
def move_sticky_note():
    """把便签移动到指定层级（desk/drawer/archive）。容量满了不拦截，只在返回值里带提醒标志。"""
    if not _check_chat_auth(request):
        return jsonify({"ok": False, "error": "unauthorized"}), 401
    data = request.json or {}
    note_id = data.get("id")
    target_status = data.get("target_status")
    if not note_id or target_status not in ("desk", "drawer", "archive"):
        return jsonify({"ok": False, "error": "缺少id，或target_status不是desk/drawer/archive之一"}), 400
    try:
        move_sticky_note_row(note_id, target_status)
        return jsonify({
            "ok": True,
            "moved_id": note_id,
            "target_status": target_status,
            "target_full": is_sticky_layer_full(target_status)
        })
    except Exception as e:
        log_error("move_sticky_note", e)
        return jsonify({"ok": False, "error": str(e)}), 500


@app.route("/api/sticky-notes/star", methods=["POST"])
def star_sticky_note():
    """设置/取消便签的标星收藏状态，跟层级无关。"""
    if not _check_chat_auth(request):
        return jsonify({"ok": False, "error": "unauthorized"}), 401
    data = request.json or {}
    note_id = data.get("id")
    is_starred = data.get("is_starred")
    if not note_id or is_starred is None:
        return jsonify({"ok": False, "error": "缺少id或is_starred参数"}), 400
    try:
        star_sticky_note_row(note_id, is_starred)
        return jsonify({"ok": True, "id": note_id, "is_starred": bool(is_starred)})
    except Exception as e:
        log_error("star_sticky_note", e)
        return jsonify({"ok": False, "error": str(e)}), 500


@app.route("/api/sticky-notes/delete", methods=["POST"])
def delete_sticky_note():
    """丢弃：物理删除，不可恢复（哪一层的便签都可以用这个接口删）。"""
    if not _check_chat_auth(request):
        return jsonify({"ok": False, "error": "unauthorized"}), 401
    data = request.json or {}
    note_id = data.get("id")
    if not note_id:
        return jsonify({"ok": False, "error": "缺少id参数"}), 400
    try:
        delete_sticky_note_row(note_id)
        return jsonify({"ok": True, "deleted_id": note_id})
    except Exception as e:
        log_error("delete_sticky_note", e)
        return jsonify({"ok": False, "error": str(e)}), 500


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


@app.route("/api/chat-edit-resend", methods=["POST"])
def chat_edit_resend():
    """编辑一条已发送的用户消息并重发：真正的"分支/回滚"，不是简单地再发一条。
    行为：把这条消息内容原地改掉，同时删除它之后的所有消息（不管是这一轮的Charon回复，
    还是编辑点之后用户又追加发过的任何内容）——对话收束到编辑的这个点，然后基于截断后
    的历史重新生成一条Charon回复接上去。跟之前"编辑=把文字填回输入框再发一遍"的区别是：
    旧方式会在记录里留下原消息+新消息两条重复内容，还得手动删掉中间的旧对话；
    这个接口一次操作就把"回滚到这里、换一种说法重新说"这件事做完。
    Body: {"id": "被编辑的用户消息id", "new_content": "编辑后的新内容"}"""
    if not _check_chat_auth(request):
        return jsonify({"ok": False, "error": "unauthorized"}), 401

    data = request.json or {}
    msg_id = data.get("id")
    new_content = (data.get("new_content") or "").strip()
    if not msg_id or not new_content:
        return jsonify({"ok": False, "error": "缺少id或new_content参数"}), 400

    try:
        target = get_chat_message_row(msg_id)
        if not target:
            return jsonify({"ok": False, "error": "没找到这条消息，可能已经被删过了"}), 404
        if target.get("role") != "user":
            return jsonify({"ok": False, "error": "只能编辑用户自己发的消息"}), 400

        target_created_at = target.get("created_at", "")
        old_content = target.get("content", "")

        # 截断：删掉这条消息之后的所有对话（用户后续消息+Charon的所有回复），
        # 同步清理events里对应时间段的记录，避免Charon下次自动醒来时还惦记着已经被覆盖的旧对话。
        delete_chat_messages_after(target_created_at)
        delete_events_after(target_created_at)

        # 原地覆盖这条用户消息的内容，不新增行、不改created_at（保持它在时间线上的原有位置）
        update_chat_message_row(msg_id, new_content)

        # 同步更新events里这条消息自己的记录内容（如果存在的话），跟chat_delete的做法对称
        delete_event_row(target_created_at, old_content)
        add_event_row("chat", f"她在网页里说：{new_content}", target_created_at)

        # 用截断后的历史（不含被替换的旧内容和旧回复）构建prompt，生成新的Charon回复。
        # 编辑重发不额外加情绪分——这是"把已经说过的话修正一下"，不是一次新的互动，
        # 原消息发送时已经加过一次分，这里只读当前分数（走自然衰减），不再调用recover_mood。
        history = load_chat_history()  # 此时target这条已经是新内容了，之后的都已删除
        mood_score = apply_mood_decay()
        chat_hours_gap = get_hours_since_last_chat()
        mood_context = get_mood_context(mood_score, chat_hours_gap)

        maybe_trigger_sweet_letter(mood_context)
        maybe_trigger_longing_letter(mood_context)

        hour = datetime.now().hour
        time_context = get_time_context(hour)
        prompt = build_chat_reply_prompt(time_context, new_content, history, mood_context)
        raw = call_deepseek(prompt)
        _, reply_msg = parse_reason_message(raw)

        charon_msg_id = new_msg_id()
        add_chat_message_row(charon_msg_id, "charon", reply_msg)

        maybe_update_chat_summary(history)

        return jsonify({
            "ok": True,
            "user_msg_id": msg_id,
            "new_content": new_content,
            "reply": reply_msg,
            "charon_msg_id": charon_msg_id
        })
    except Exception as e:
        log_error("chat_edit_resend", e)
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

        # 同步写一笔events，方便活动记录里也能看到这次互动
        add_event_row("chat", f"她在网页里说：{user_message}", user_created_at)

        # 用户重新上线说话了：查岗状态清零，下次她离线重新计时
        try:
            reset_checkin_state()
        except Exception as e:
            log_error("chat_send:reset_checkin_state", e)

        # 真实聊天：情绪值+10，并刷新"上次聊天时间"（影响下次衰减计算的起点）
        old_score, mood_score = recover_mood(MOOD_RECOVERY_CHAT, mark_chat=True)
        chat_hours_gap = get_hours_since_last_chat()
        mood_context = get_mood_context(mood_score, chat_hours_gap)

        # 检查当前状态是否命中情书触发条件
        maybe_trigger_sweet_letter(mood_context)
        maybe_trigger_longing_letter(mood_context)

        # 生成Charon的回应，带上历史让语气能接得上
        hour = datetime.now().hour
        time_context = get_time_context(hour)

        prompt = build_chat_reply_prompt(time_context, user_message, history, mood_context)
        raw = call_deepseek(prompt)
        _, reply_msg = parse_reason_message(raw)

        charon_msg_id = new_msg_id()
        add_chat_message_row(charon_msg_id, "charon", reply_msg)

        # 顺手检查一下要不要更新滚动摘要（只在对话变长之后才会真正触发，不影响每次的响应速度）
        maybe_update_chat_summary(history)

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
        chat_hours_gap = get_hours_since_last_chat()
        mood_score = apply_mood_decay()
        mood_context = get_mood_context(mood_score, chat_hours_gap)

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


@app.route("/api/persona", methods=["GET"])
def get_persona():
    """读取当前人设/长期记忆文本，给 /persona 页面加载用。"""
    if not _check_chat_auth(request):
        return jsonify({"ok": False, "error": "unauthorized"}), 401
    return jsonify({"ok": True, "text": load_persona_memory()})


@app.route("/api/persona", methods=["POST"])
def set_persona():
    """保存人设/长期记忆文本。整体覆盖写入，改完之后所有prompt立刻生效，不用重新部署。"""
    if not _check_chat_auth(request):
        return jsonify({"ok": False, "error": "unauthorized"}), 401
    data = request.json or {}
    text = data.get("text", "").strip()
    if not text:
        return jsonify({"ok": False, "error": "内容不能为空"}), 400
    try:
        save_persona_memory(text)
        return jsonify({"ok": True, "text": text})
    except Exception as e:
        log_error("set_persona", e)
        return jsonify({"ok": False, "error": str(e)}), 500


@app.route("/persona", methods=["GET"])
def persona_page():
    """独立的人设编辑小页面，跟/chat完全分开、不共用模板，方便随时改Charon的长期记忆/性格设定文字。
    改完点保存，立刻生效，不用等重新部署。"""
    if not _check_chat_auth(request):
        return "<h3>需要访问口令</h3><p>在链接后加 ?code=你的口令</p>", 401

    code_param = request.args.get("code", "")

    return f"""<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Charon 人设编辑</title>
<style>
  body {{
    font-family: -apple-system, "PingFang SC", sans-serif;
    background: #faf8f5;
    color: #333;
    max-width: 640px;
    margin: 0 auto;
    padding: 24px 16px 60px;
  }}
  h2 {{ font-size: 18px; font-weight: 600; margin-bottom: 4px; }}
  p.hint {{ color: #999; font-size: 13px; margin-top: 0; margin-bottom: 20px; }}
  textarea {{
    width: 100%;
    min-height: 320px;
    box-sizing: border-box;
    padding: 14px;
    font-size: 15px;
    line-height: 1.6;
    border: 1px solid #ddd;
    border-radius: 10px;
    resize: vertical;
    font-family: inherit;
  }}
  button {{
    margin-top: 14px;
    padding: 10px 22px;
    background: #333;
    color: #fff;
    border: none;
    border-radius: 8px;
    font-size: 15px;
    cursor: pointer;
  }}
  button:disabled {{ opacity: 0.5; }}
  #status {{ margin-left: 12px; font-size: 13px; color: #4a934a; }}
  #error {{ margin-left: 12px; font-size: 13px; color: #c0392b; }}
</style>
</head>
<body>
  <h2>Charon 人设 / 长期记忆</h2>
  <p class="hint">这段文字会带进Charon每一次回复的生成里。改完点保存，立刻生效，不用等重新部署。</p>
  <textarea id="text"></textarea>
  <br>
  <button id="saveBtn" onclick="save()">保存</button>
  <span id="status"></span>
  <span id="error"></span>

<script>
  const codeParam = {code_param!r};

  async function load() {{
    const res = await fetch(`/api/persona?code=${{encodeURIComponent(codeParam)}}`);
    const data = await res.json();
    if (data.ok) {{
      document.getElementById('text').value = data.text;
    }} else {{
      document.getElementById('error').textContent = '加载失败：' + (data.error || '未知错误');
    }}
  }}

  async function save() {{
    const btn = document.getElementById('saveBtn');
    const statusEl = document.getElementById('status');
    const errorEl = document.getElementById('error');
    statusEl.textContent = '';
    errorEl.textContent = '';
    btn.disabled = true;
    try {{
      const res = await fetch(`/api/persona?code=${{encodeURIComponent(codeParam)}}`, {{
        method: 'POST',
        headers: {{'Content-Type': 'application/json'}},
        body: JSON.stringify({{text: document.getElementById('text').value}})
      }});
      const data = await res.json();
      if (data.ok) {{
        statusEl.textContent = '已保存';
        setTimeout(() => statusEl.textContent = '', 2000);
      }} else {{
        errorEl.textContent = '保存失败：' + (data.error || '未知错误');
      }}
    }} catch (e) {{
      errorEl.textContent = '保存失败：网络错误';
    }} finally {{
      btn.disabled = false;
    }}
  }}

  load();
</script>
</body>
</html>"""


@app.route("/manifest.json", methods=["GET"])
def pwa_manifest():
    """PWA清单文件，iOS Safari"添加到主屏幕"时读取这个来决定图标/名字/启动页。
    display设为standalone，去掉浏览器地址栏，看起来像原生App，这是能收Web Push的前提之一。"""
    manifest = {
        "name": "Charon",
        "short_name": "Charon",
        "start_url": "/chat",
        "display": "standalone",
        "background_color": "#fbf3f5",
        "theme_color": "#fbf3f5",
        "icons": [
            {"src": APP_ICON_192, "sizes": "192x192", "type": "image/jpeg"},
            {"src": APP_ICON_512, "sizes": "512x512", "type": "image/jpeg"},
        ],
    }
    return jsonify(manifest)


@app.route("/service-worker.js", methods=["GET"])
def service_worker():
    """Service Worker脚本，必须从根路径提供才能控制整个站点的作用域。
    职责很单一：收到push事件就弹通知；用户点通知就把浏览器/PWA焦点切到chat页面
    （如果已经开着就聚焦那个tab，没开就新开一个），这样点通知能直接跳转到网页。"""
    sw_code = """
self.addEventListener('push', function(event) {
  let data = {title: 'Charon', body: '', icon: '', url: '/chat'};
  try { data = event.data.json(); } catch (e) {}
  event.waitUntil(
    self.registration.showNotification(data.title || 'Charon', {
      body: data.body || '',
      icon: data.icon || '',
      badge: data.icon || '',
      data: { url: data.url || '/chat' }
    })
  );
});

self.addEventListener('notificationclick', function(event) {
  event.notification.close();
  const targetUrl = event.notification.data && event.notification.data.url ? event.notification.data.url : '/chat';
  event.waitUntil(
    clients.matchAll({ type: 'window', includeUncontrolled: true }).then(function(clientList) {
      for (const client of clientList) {
        if (client.url.includes('/chat') && 'focus' in client) {
          return client.focus();
        }
      }
      if (clients.openWindow) {
        return clients.openWindow(targetUrl);
      }
    })
  );
});
"""
    return app.response_class(sw_code, mimetype="application/javascript")


@app.route("/api/push/vapid-public-key", methods=["GET"])
def push_vapid_public_key():
    """前端订阅时需要用这个公钥构造 applicationServerKey，不涉及隐私数据，不需要鉴权。"""
    if not VAPID_PUBLIC_KEY:
        return jsonify({"ok": False, "error": "VAPID_PUBLIC_KEY 未配置"}), 500
    return jsonify({"ok": True, "public_key": VAPID_PUBLIC_KEY})


@app.route("/api/push/subscribe", methods=["POST"])
def push_subscribe():
    """前端拿到浏览器的PushSubscription对象后POST到这里存起来。"""
    if not _check_chat_auth(request):
        return jsonify({"ok": False, "error": "unauthorized"}), 401
    data = request.json or {}
    endpoint = data.get("endpoint")
    keys = data.get("keys") or {}
    p256dh = keys.get("p256dh")
    auth = keys.get("auth")
    if not endpoint or not p256dh or not auth:
        return jsonify({"ok": False, "error": "订阅信息不完整"}), 400
    try:
        save_push_subscription(endpoint, p256dh, auth)
        return jsonify({"ok": True})
    except Exception as e:
        log_error("push_subscribe", e)
        return jsonify({"ok": False, "error": str(e)}), 500


@app.route("/api/push/unsubscribe", methods=["POST"])
def push_unsubscribe():
    """设备主动取消订阅时调用，把对应endpoint从数据库删掉。"""
    if not _check_chat_auth(request):
        return jsonify({"ok": False, "error": "unauthorized"}), 401
    data = request.json or {}
    endpoint = data.get("endpoint")
    if not endpoint:
        return jsonify({"ok": False, "error": "缺少endpoint"}), 400
    try:
        delete_push_subscription(endpoint)
        return jsonify({"ok": True})
    except Exception as e:
        log_error("push_unsubscribe", e)
        return jsonify({"ok": False, "error": str(e)}), 500


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

    resp = app.make_response(render_template(
        "chat.html",
        avatar_charon=CHAT_AVATAR_CHARON,
        avatar_user=CHAT_AVATAR_USER,
        code_param=code_param,
    ))

    # 只要这次是靠URL参数里的code校验通过的，就顺手把它写进Cookie（一年有效期）。
    # 这样"添加到主屏幕"生成的PWA快捷方式，即便固定用的是不带code的URL启动，
    # 之后也能靠这个Cookie自动放行，不会永久卡在口令页。
    # httponly=False是必须的：Service Worker和前端fetch都可能需要读取/依赖这个状态，
    # 且这不是敏感的登录凭证系统，只是个人使用的轻量访问口令，风险可接受。
    if CHAT_ACCESS_CODE and code_param == CHAT_ACCESS_CODE:
        resp.set_cookie("chat_code", CHAT_ACCESS_CODE, max_age=365 * 24 * 3600, samesite="Lax")

    return resp


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
    """后台循环：不再固定55分钟发一次消息，改成高频（5分钟）地"看一眼"要不要触发查岗。
    真正发消息的频率完全由 check_and_run_checkin() 里的状态机决定。"""
    while True:
        try:
            check_and_run_checkin()
        except Exception as e:
            log_error("keepalive", e)
        time.sleep(CHECKIN_LOOP_INTERVAL)


if __name__ == "__main__":
    t = threading.Thread(target=keepalive, daemon=True)
    t.start()
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
