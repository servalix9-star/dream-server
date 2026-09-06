"""
core.py —— 共享基础设施层

拆分说明：main.py 原本是一个3500多行的单文件，随着朋友圈/情书/便签/意图识别等
功能陆续加入，"每次改动都要通读整个文件"的成本越来越高。这次拆分把真正跨功能
共用的基础设施（不属于任何单一业务功能，而是所有功能都依赖的"地基"）搬到这里：

  - Flask app 实例、环境变量、Supabase连接
  - _supabase_request：数据库读写的统一入口
  - get_app_config / set_app_config：通用配置表读写（mood/model_config/persona_memory等都用它）
  - 模型注册表 + 调用系统：get_current_model / call_deepseek / call_model_stream 等
  - log_error：统一错误日志
  - events 表核心读写：add_event_row / load_events（这是让聊天/便签/情书/朋友圈
    互相"看到"对方发生了什么的核心枢纽，很多模块都要用它同步事件）
  - mood 情绪值状态机：所有依赖心境的功能（便签语气、情书触发、朋友圈AI互动）都读这里
  - _extract_json_field：解析模型JSON输出的小工具，多处业务逻辑公用
  - persona_memory：人设/长期记忆读写
  - _check_chat_auth：访问口令校验

main.py 和 moments.py（以及未来拆出的 letters.py / notes.py 等）都从这里
`from core import xxx`，core.py 本身不反向依赖它们，避免循环导入。
"""

import sys
# 强制stdout/stderr无缓冲：Render等容器化平台运行时，Python检测到stdout不是终端会自动切换成
# 块缓冲（block buffering），导致print()内容一直攒在内存里不实时写出，甚至长期看不到。
# 这里在最开头就重新包装一次，保证后面所有print()都是行缓冲、立刻可见，不用每个print单独加flush=True。
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

import subprocess
subprocess.run(["pip", "install", "requests", "pywebpush"], capture_output=True)

from flask import Flask, request, jsonify, render_template, Response
from datetime import datetime, date
import json, os, requests, threading, time, traceback, random

# 显式指定模板/静态资源根目录为本文件所在目录，不依赖"app恰好和templates/
# 放在同一目录"这种隐式约定——现在app对象定义在core.py里，但真正的
# templates/文件夹是跟main.py（以及未来的其他业务模块）一起部署的，
# 两者路径应该一致，这里用__file__显式钉死，避免以后目录结构变动时
# render_template莫名其妙找不到模板。
app = Flask(__name__, root_path=os.path.dirname(os.path.abspath(__file__)))

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
# gemai-* 系列是接入的 gemai.cc 代理站模型，纯粹作为备选，走独立的供应商配置（见 DEFAULT_MODEL_REGISTRY）。
# 这类代理站的具体渠道时常变动（之前接的[官逆]gemini-2.5-pro出现过503 model_not_found，渠道下线），
# 所以这里一次接入9个当前指定的型号，覆盖GPT/Gemini/Grok三个系列，哪个能用切哪个。
# 其中gemini-2.5-pro有[满血A][满血D]两条渠道，gemini-3.1-pro-preview有[官逆]和[满血A]+thinking两条渠道，
# 内部id用 -a / -d / -thinking 后缀区分，real_model原样保留完整前缀标注（渠道识别用）。
# ==========================================================================
# 模型注册表（动态配置管理）
# ==========================================================================
# 历史上这里是三个平行的硬编码字典（AVAILABLE_MODELS / MODEL_THINKING_MAP /
# MODEL_PROVIDER_MAP），每次某个gemai.cc代理站渠道挂了/换了，都要改代码重新部署。
#
# 现在改成：这份字典只是"出厂默认值"（DEFAULT_MODEL_REGISTRY），真正生效的配置
# 优先从 Supabase app_config 表的 model_registry 这个key读取（见下面get_model_registry）。
# 数据库里没配置过时，自动回退到这份默认值，保证第一次上线/数据库还没初始化时不会挂。
#
# 结构：每个模型id对应一条完整配置：
#   - active: 是否启用。false的模型不会出现在前端下拉菜单，也不能被选中。
#     公益站渠道挂了，不用改代码，直接去Supabase把对应条目的active改成false即可。
#   - base_url: 接口地址
#   - api_key_env: 该用哪个环境变量的值作为api_key（不直接存密钥本身，密钥仍然
#     只放在Render环境变量里；这样即使Supabase数据泄露，密钥也不会跟着泄露）。
#   - real_model: 发给上游时真正用的模型名（代理站渠道识别用，前缀方括号必须原样保留）
#   - supports_thinking: 是否要在请求体里带DeepSeek风格的thinking字段
#   - thinking: 该模型的思考模式（disabled/enabled），仅supports_thinking=True时生效
#   - api_style: "openai_compatible"（默认，DeepSeek官方/gemai.cc代理站都是这种messages结构）
#     或 "gemini_native"（Google官方原生接口，contents/parts结构，key走x-goog-api-key header）
#
# 新增模型/供应商：不用改代码，直接去Supabase的app_config表编辑model_registry这条JSON即可，
# 改完最多60秒生效（见MODEL_REGISTRY_TTL缓存）。
DEFAULT_MODEL_REGISTRY = {
    "deepseek-v4-flash": {
        "active": True,
        "base_url": "https://api.deepseek.com/chat/completions",
        "api_key_env": "DEEPSEEK_API_KEY",
        "real_model": "deepseek-v4-flash",
        "supports_thinking": True,
        "thinking": "disabled",
    },
    "deepseek-v4-pro": {
        "active": True,
        "base_url": "https://api.deepseek.com/chat/completions",
        "api_key_env": "DEEPSEEK_API_KEY",
        "real_model": "deepseek-v4-pro",
        "supports_thinking": True,
        "thinking": "enabled",
    },
    "gemai-gpt-4o-mini": {
        "active": True,
        "base_url": "https://api.gemai.cc/v1/chat/completions",
        "api_key_env": "GEMAI_API_KEY",
        "real_model": "[官逆]gpt-4o-mini",  # 官逆渠道
        "supports_thinking": False,
        "thinking": "disabled",
    },
    "gemai-gpt-4.1-mini": {
        "active": True,
        "base_url": "https://api.gemai.cc/v1/chat/completions",
        "api_key_env": "GEMAI_API_KEY",
        "real_model": "[官逆]gpt-4.1-mini",  # 官逆渠道
        "supports_thinking": False,
        "thinking": "disabled",
    },
    "gemai-gpt-5-mini": {
        "active": True,
        "base_url": "https://api.gemai.cc/v1/chat/completions",
        "api_key_env": "GEMAI_API_KEY",
        "real_model": "[官逆]gpt-5-mini",  # 官逆渠道
        "supports_thinking": False,
        "thinking": "disabled",
    },
    "gemai-gemini-2.5-flash-a": {
        "active": True,
        "base_url": "https://api.gemai.cc/v1/chat/completions",
        "api_key_env": "GEMAI_API_KEY",
        "real_model": "[满血A]gemini-2.5-flash",  # 满血A渠道
        "supports_thinking": False,
        "thinking": "disabled",
    },
    "gemai-gemini-2.5-pro-a": {
        "active": True,
        "base_url": "https://api.gemai.cc/v1/chat/completions",
        "api_key_env": "GEMAI_API_KEY",
        "real_model": "[满血A]gemini-2.5-pro",  # 满血A渠道
        "supports_thinking": False,
        "thinking": "disabled",
    },
    "gemai-gemini-2.5-pro-d": {
        "active": True,
        "base_url": "https://api.gemai.cc/v1/chat/completions",
        "api_key_env": "GEMAI_API_KEY",
        "real_model": "[满血D]gemini-2.5-pro",  # 满血D渠道
        "supports_thinking": False,
        "thinking": "disabled",
    },
    "gemai-gemini-3.1-pro": {
        "active": True,
        "base_url": "https://api.gemai.cc/v1/chat/completions",
        "api_key_env": "GEMAI_API_KEY",
        "real_model": "[官逆]gemini-3.1-pro-preview",  # 官逆渠道
        "supports_thinking": False,
        "thinking": "disabled",
    },
    "gemai-gemini-3.1-pro-thinking": {
        "active": True,
        "base_url": "https://api.gemai.cc/v1/chat/completions",
        "api_key_env": "GEMAI_API_KEY",
        "real_model": "[满血A]gemini-3.1-pro-preview-thinking-128",  # 满血A渠道，开启深度思考
        "supports_thinking": False,
        "thinking": "disabled",
    },
    "gemai-grok-4": {
        "active": True,
        "base_url": "https://api.gemai.cc/v1/chat/completions",
        "api_key_env": "GEMAI_API_KEY",
        "real_model": "grok-4",  # 无前缀标注
        "supports_thinking": False,
        "thinking": "disabled",
    },
    # Google官方Gemini API（AI Studio申请的key），走原生Gemini接口。
    # 注意：2026年Google把AI Studio新发的key格式从AIza换成了AQ.，
    # AQ.格式key在OpenAI兼容端点（/v1beta/openai/chat/completions）会返回401，
    # 但在原生端点（generativelanguage.googleapis.com，用x-goog-api-key header传key）工作正常，
    # 所以这几个模型都走api_style=gemini_native，不能用openai_compatible的payload格式。
    #
    # 之前实测gemini-official-flash（用gemini-3.6-flash）连接完全正常，只是有一次被判定为
    # PROHIBITED_CONTENT拦截，怀疑是对话内容触发了默认的安全过滤级别。现在在_call_model_raw里
    # 给gemini_native分支统一加了safety_settings（四个类别都设为BLOCK_NONE，见下方GEMINI_SAFETY_SETTINGS），
    # 尝试放宽过滤。需要说明：Google对"色情内容"这一类别的过滤，即使设了BLOCK_NONE，
    # 在某些情况下也不保证完全不拦截（这是Google侧的策略，不是代码能完全控制的），
    # 所以这几个模型仍建议留一个非Gemini的备选，别完全依赖它们。
    #
    # pro系列（gemini-3.1-pro-preview）之前实测在免费层配额为0（quota limit: 0），
    # 需要项目开通计费才能用，这里继续保持关闭。
    "gemini-official-flash": {
        "active": True,
        "base_url": "https://generativelanguage.googleapis.com/v1beta/models/gemini-3.6-flash:generateContent",
        "api_key_env": "GEMINI_API_KEY",
        "real_model": "gemini-3.6-flash",
        "supports_thinking": False,
        "thinking": "disabled",
        "api_style": "gemini_native",
    },
    "gemini-3.5-flash": {
        "active": True,
        "base_url": "https://generativelanguage.googleapis.com/v1beta/models/gemini-3.5-flash:generateContent",
        "api_key_env": "GEMINI_API_KEY",
        "real_model": "gemini-3.5-flash",
        "supports_thinking": False,
        "thinking": "disabled",
        "api_style": "gemini_native",
    },
    "gemini-3.5-flash-lite": {
        "active": True,
        "base_url": "https://generativelanguage.googleapis.com/v1beta/models/gemini-3.5-flash-lite:generateContent",
        "api_key_env": "GEMINI_API_KEY",
        "real_model": "gemini-3.5-flash-lite",
        "supports_thinking": False,
        "thinking": "disabled",
        "api_style": "gemini_native",
    },
    "gemini-official-pro": {
        "active": False,
        "base_url": "https://generativelanguage.googleapis.com/v1beta/models/gemini-3.1-pro-preview:generateContent",
        "api_key_env": "GEMINI_API_KEY",
        "real_model": "gemini-3.1-pro-preview",
        "supports_thinking": False,
        "thinking": "disabled",
        "api_style": "gemini_native",
    },
}

# Gemini原生接口的安全过滤设置：四个类别统一设为BLOCK_NONE（不拦截）。
# 说明：Google对HARM_CATEGORY_SEXUALLY_EXPLICIT这一类的过滤，即使设了BLOCK_NONE，
# 也不保证在所有情况下都完全放行——这是Google服务端策略决定的，代码层面能做的只有这么多。
GEMINI_SAFETY_SETTINGS = [
    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
]

DEFAULT_MODEL = "deepseek-v4-flash"

# model_registry从Supabase读出来后缓存在内存里，避免每次对话都查一次数据库。
# TTL设置得短（60秒），改了配置不用重启服务，最多等1分钟就生效。
MODEL_REGISTRY_TTL = 60
_model_registry_cache = {"data": None, "at": 0}


def get_model_registry():
    """获取当前生效的模型注册表：优先读Supabase app_config表的model_registry这个key，
    没配置过（或读取失败）就回退到DEFAULT_MODEL_REGISTRY，保证不会因为数据库问题导致模型全部不可用。
    带60秒内存缓存，避免每次call_deepseek/查询可用模型列表都打一次Supabase。"""
    now = time.time()
    if _model_registry_cache["data"] is None or now - _model_registry_cache["at"] > MODEL_REGISTRY_TTL:
        _model_registry_cache["data"] = get_app_config("model_registry", DEFAULT_MODEL_REGISTRY)
        _model_registry_cache["at"] = now
    return _model_registry_cache["data"]


def get_available_models():
    """返回当前active=true的模型id列表，按注册表里的原始顺序，供前端下拉菜单展示。"""
    registry = get_model_registry()
    return [mid for mid, cfg in registry.items() if cfg.get("active")]


def resolve_api_key(cfg):
    """从模型配置里的api_key_env字段，读出对应环境变量的真实密钥值。
    数据库里只存环境变量名字（比如"GEMAI_API_KEY"），不存密钥明文本身，
    这样即使Supabase权限设置疏漏导致数据被看到，密钥依然安全，只有Render后台能看到真实值。"""
    env_name = cfg.get("api_key_env")
    if not env_name:
        return None
    return os.environ.get(env_name)


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
    存服务端而不是浏览器本地，这样换设备打开聊天页选择依然一致。
    这里校验用的是当前生效的注册表（get_available_models，只含active=true的模型），
    如果之前选中的模型后来被停用了，会自动回退到DEFAULT_MODEL，不会调用一个已下线的渠道。"""
    data = get_app_config("model_config", {"model": DEFAULT_MODEL})
    model = data.get("model") if isinstance(data, dict) else None
    if model in get_available_models():
        return model
    return DEFAULT_MODEL


def get_thinking_config():
    """根据当前选中的模型返回对应的thinking参数。
    flash用disabled保持快速直接、且temperature等参数生效；
    pro用enabled真正发挥深度推理能力（此时temperature等参数会被静默忽略，这是预期代价）。"""
    model = get_current_model()
    cfg = get_model_registry().get(model, {})
    thinking_type = cfg.get("thinking", "disabled")
    return {"type": thinking_type}


def set_current_model(model):
    if model not in get_available_models():
        raise ValueError(f"不支持的模型: {model}")
    set_app_config("model_config", {"model": model})

def log_error(context, e):
    line = f"{datetime.now().isoformat()} [{context}] {e}\n{traceback.format_exc()}\n"
    print(line)
    try:
        with open(ERROR_LOG, "a") as f:
            f.write(line)
    except Exception:
        pass


# ---- events 表核心读写（跨模块同步事件用） ----

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



def _normalize_to_messages(prompt_or_messages):
    """统一入参：老调用点传的是一整段字符串prompt（单轮场景，比如主动消息、便签、摘要生成），
    新调用点（多轮聊天回复场景）传的是[{"role": "user"/"assistant", "content": "..."}]结构。
    这里统一转成openai风格的messages列表，方便下面两个分支共用同一份逻辑。
    单条字符串会被包成一条user消息——行为跟改动前完全一致，不影响其余调用点。"""
    if isinstance(prompt_or_messages, str):
        return [{"role": "user", "content": prompt_or_messages}]
    return prompt_or_messages


def _call_model_raw(prompt_or_messages):
    """真正干活的模型调用。
    改名是因为外层现在包了一层健康记录（call_deepseek），这个函数只管发请求拿结果，
    成功还是失败都不管，交给外层统一记账。

    参数现在既可以是字符串（老用法，单轮，自动包成一条user消息），
    也可以是messages列表（新用法，真正的多轮对话结构：[{"role":"user"/"assistant","content":...}, ...]）。
    这是修复"上下文能力差"的关键改动：之前不管传多少历史，最终都被拼接成一段
    文本塞进唯一一条user消息里发给模型，模型看到的永远是单轮"续写剧本"任务，
    完全没用上它自己原生的多轮对话理解能力——这正是"简单的话能接上，稍微复杂点
    就答非所问"的根源。现在改成真正按轮次构造messages/contents，模型才能像
    正常聊天那样，理解"你问了A，我答了B，你现在追问C"这种指代和逻辑链条。"""
    messages = _normalize_to_messages(prompt_or_messages)

    model = get_current_model()
    provider = get_model_registry().get(model)
    if not provider:
        raise RuntimeError(f"模型 {model} 没有配置对应的供应商信息")
    api_key = resolve_api_key(provider)
    if not api_key:
        raise RuntimeError(f"模型 {model} 对应的 API key 未设置（环境变量缺失：{provider.get('api_key_env')}）")

    # api_style默认是openai_compatible（DeepSeek官方 / gemai.cc代理站都是这种，
    # messages结构 + Authorization: Bearer头）。Gemini官方原生接口结构不同，
    # 单独分流处理，不污染现有格式的调用路径。
    api_style = provider.get("api_style", "openai_compatible")

    if api_style == "gemini_native":
        # Gemini原生接口的"contents"数组只放user/model两种轮次，
        # 每一轮是{"role": "user"/"model", "parts": [...]}——它的assistant角色叫"model"。
        # system角色的内容不能混进contents当普通轮次（那样等于让人设被误当成"用户说的话"，
        # 权重和语义都不对），要单独走systemInstruction字段，这是Gemini官方推荐的做法。
        system_texts = [m["content"] for m in messages if m["role"] == "system"]
        contents = [
            {
                "role": "model" if m["role"] == "assistant" else "user",
                "parts": [{"text": m["content"]}],
            }
            for m in messages
            if m["role"] != "system"
        ]
        payload = {
            "contents": contents,
            # 温度从1.2降到1.0：1.2偏高，容易让语言变得跳脱、甚至偏离人设，
            # 1.0是更常见的"有个性但不失控"区间，可以按实际效果再微调。
            "generationConfig": {"temperature": 1.0},
            "safetySettings": GEMINI_SAFETY_SETTINGS,
        }
        if system_texts:
            payload["systemInstruction"] = {"parts": [{"text": "\n\n".join(system_texts)}]}
        resp = requests.post(
            provider["base_url"],
            headers={
                "x-goog-api-key": api_key,
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

    # ---- 以下是openai_compatible分支：DeepSeek官方 / gemai.cc代理站都走这条 ----
    # 直接把messages原样传过去：多轮聊天场景下这就是真正的user/assistant交替结构，
    # 老的单轮调用点下就是原来的[{"role": "user", "content": prompt}]，行为完全不变。
    payload = {
        "model": provider["real_model"],
        "messages": messages,
        "temperature": 1.0,
    }
    if provider["supports_thinking"]:
        payload["thinking"] = get_thinking_config()

    resp = requests.post(
        provider["base_url"],
        headers={
            "Authorization": f"Bearer {api_key}",
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


def call_model_stream(messages):
    """流式版模型调用：逐块yield文本片段，供聊天回复场景实现"打字机效果"用。

    只用于网页实时对话（/api/chat-send等），且只接受messages列表（多轮结构），
    不兼容老的字符串prompt用法——因为流式场景下模型直接输出纯对话内容，
    不再包一层{reason, message}的JSON（JSON必须等完整生成完才能解析，
    没法一边流一边显示，这正是要做真流式必须去掉JSON包裹的原因）。

    调用方在生成器耗尽后可以读its .final_text / .error 属性拿到完整结果和错误信息
    （通过闭包变量实现，见下方chat_send里的用法）。

    异常处理：网络请求本身失败会在第一次yield之前抛出，调用方需要用try/except包住
    对这个生成器的遍历；如果是在流式过程中途断线，会尽量把已经收到的部分作为
    最终结果返回，不会让用户已经看到的文字凭空消失。
    """
    model = get_current_model()
    provider = get_model_registry().get(model)
    if not provider:
        raise RuntimeError(f"模型 {model} 没有配置对应的供应商信息")
    api_key = resolve_api_key(provider)
    if not api_key:
        raise RuntimeError(f"模型 {model} 对应的 API key 未设置（环境变量缺失：{provider.get('api_key_env')}）")

    api_style = provider.get("api_style", "openai_compatible")

    if api_style == "gemini_native":
        system_texts = [m["content"] for m in messages if m["role"] == "system"]
        contents = [
            {
                "role": "model" if m["role"] == "assistant" else "user",
                "parts": [{"text": m["content"]}],
            }
            for m in messages
            if m["role"] != "system"
        ]
        payload = {
            "contents": contents,
            "generationConfig": {"temperature": 1.0},
            "safetySettings": GEMINI_SAFETY_SETTINGS,
        }
        if system_texts:
            payload["systemInstruction"] = {"parts": [{"text": "\n\n".join(system_texts)}]}

        # Gemini原生的流式端点：把:generateContent换成:streamGenerateContent，
        # 并加?alt=sse让它按SSE格式（data: {...}\n\n）逐块推送，而不是一次性返回大JSON数组。
        stream_url = provider["base_url"].replace(":generateContent", ":streamGenerateContent") + "?alt=sse"
        resp = requests.post(
            stream_url,
            headers={"x-goog-api-key": api_key, "Content-Type": "application/json"},
            json=payload,
            stream=True,
            timeout=60,
        )
        if resp.status_code != 200:
            raise RuntimeError(f"模型API error: model={model} status={resp.status_code} body={resp.text}")

        # 注意：这里不能用 iter_lines(decode_unicode=True)。SSE是分块(chunked)传输，
        # decode_unicode=True 依赖 requests 对 resp.encoding 的猜测（猜不到就退化成
        # ISO-8859-1），而且是按网络包边界解码，一个多字节UTF-8字符（如中文，3字节）
        # 如果恰好被切在两个包之间，就会在行内部被提前、错误地解码，导致乱码——
        # 这正是"偶尔乱码、重试才正常"这种随机性表现的根本原因（命中网络分包时机才炸）。
        # 改成按原始字节迭代 + 手动UTF-8解码，自动跨块缓冲不完整的字节序列，彻底规避这个问题。
        buffer = b""
        for raw_chunk in resp.iter_content(chunk_size=None):
            if not raw_chunk:
                continue
            buffer += raw_chunk
            while b"\n" in buffer:
                raw_line, buffer = buffer.split(b"\n", 1)
                try:
                    line = raw_line.decode("utf-8")
                except UnicodeDecodeError:
                    # 说明多字节字符被切断在了buffer末尾，把这半截数据放回buffer继续等下一块拼完整
                    buffer = raw_line + b"\n" + buffer
                    break
                line = line.rstrip("\r")
                if not line or not line.startswith("data: "):
                    continue
                data_str = line[len("data: "):].strip()
                if not data_str or data_str == "[DONE]":
                    continue
                try:
                    chunk = json.loads(data_str)
                    candidates = chunk.get("candidates") or []
                    if not candidates:
                        continue
                    parts = candidates[0].get("content", {}).get("parts", [])
                    text = "".join(p.get("text", "") for p in parts)
                    if text:
                        yield text
                except (json.JSONDecodeError, KeyError, IndexError):
                    continue
        return

    # ---- openai_compatible分支：DeepSeek官方 / gemai.cc代理站，标准SSE格式 ----
    payload = {
        "model": provider["real_model"],
        "messages": messages,
        "temperature": 1.0,
        "stream": True,
    }
    if provider["supports_thinking"]:
        payload["thinking"] = get_thinking_config()

    resp = requests.post(
        provider["base_url"],
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json=payload,
        stream=True,
        timeout=60,
    )
    if resp.status_code != 200:
        raise RuntimeError(f"模型API error: model={model} status={resp.status_code} body={resp.text}")

    # 同上：不用decode_unicode=True，理由见gemini_native分支里的注释。
    # gemai.cc是中转代理，多加了一层转发，更容易在分包时机上踩中这个问题。
    buffer = b""
    for raw_chunk in resp.iter_content(chunk_size=None):
        if not raw_chunk:
            continue
        buffer += raw_chunk
        while b"\n" in buffer:
            raw_line, buffer = buffer.split(b"\n", 1)
            try:
                line = raw_line.decode("utf-8")
            except UnicodeDecodeError:
                buffer = raw_line + b"\n" + buffer
                break
            line = line.rstrip("\r")
            if not line or not line.startswith("data: "):
                continue
            data_str = line[len("data: "):].strip()
            if not data_str or data_str == "[DONE]":
                continue
            try:
                chunk = json.loads(data_str)
                choices = chunk.get("choices") or []
                if not choices:
                    continue
                delta = choices[0].get("delta", {})
                text = delta.get("content", "")
                if text:
                    yield text
            except (json.JSONDecodeError, KeyError, IndexError):
                continue


# 连续失败达到这个次数，就在前端菜单里把这个模型标记为"不健康"（红点）。
# 不是失败一次就标红，是为了避免网络抖动这种偶发问题就被误判为模型挂了。
MODEL_UNHEALTHY_THRESHOLD = 3


def get_model_health():
    """读取所有模型的健康记录：{model_id: {"consecutive_failures": int, "last_error": str,
    "last_success_at": str, "last_failure_at": str}}。跟model_registry一样存在app_config表里，
    key叫model_health，没有记录的模型视为"健康"（毕竟还没调用过，谈不上坏）。"""
    return get_app_config("model_health", {})


def _record_model_result(model, success, error_text=None):
    """每次call_deepseek调用结束（不管成功失败）都记一笔，用于前端菜单显示健康状态。
    成功：把这个模型的连续失败次数清零。
    失败：连续失败次数+1，同时记下最新一次的错误信息，方便你在状态页面里看出个大概原因。
    这里用try/except包起来且不重新抛出：记账逻辑本身出问题，不应该影响真正的模型调用结果。"""
    try:
        health = get_app_config("model_health", {})
        entry = health.get(model, {"consecutive_failures": 0})
        now_str = datetime.now().isoformat()
        if success:
            entry["consecutive_failures"] = 0
            entry["last_success_at"] = now_str
        else:
            entry["consecutive_failures"] = entry.get("consecutive_failures", 0) + 1
            entry["last_failure_at"] = now_str
            # 错误信息可能很长（比如完整的API报错JSON），只截取前200字，
            # 够看出个大概原因（401/429/模型下线之类），不需要存全文。
            entry["last_error"] = (error_text or "")[:200]
        health[model] = entry
        set_app_config("model_health", health)
    except Exception as e:
        log_error("_record_model_result", e)


def call_deepseek(prompt_or_messages):
    """对外接口不变，函数名和调用方式跟以前完全一样（历史原因保留这个名字）。
    现在prompt_or_messages既可以是字符串（老用法，单轮），也可以是messages列表
    （新用法，多轮聊天场景，见build_chat_messages）——具体透传给_call_model_raw处理。
    这里只是加了一层健康记录：调用_call_model_raw()真正发请求，
    成功就清零这个模型的失败计数，失败就+1并记下错误原因，供前端菜单显示红绿点用。"""
    model = get_current_model()
    try:
        result = _call_model_raw(prompt_or_messages)
        _record_model_result(model, success=True)
        return result
    except Exception as e:
        _record_model_result(model, success=False, error_text=str(e))
        raise



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
