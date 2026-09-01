import subprocess
subprocess.run(["pip", "install", "requests"], capture_output=True)

from flask import Flask, request, jsonify, render_template
from datetime import datetime, date
import json, os, requests, threading, time, traceback, random

app = Flask(__name__)

# ---- 数据持久化：Supabase（PostgREST） ----
SUPABASE_URL = os.environ.get("SUPABASE_URL", "").rstrip("/")
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
CHAT_ACCESS_CODE = os.environ.get("CHAT_ACCESS_CODE")

# DeepSeek 模型与思考参数联动配置
AVAILABLE_MODELS = ["deepseek-v4-flash", "deepseek-v4-pro"]
DEFAULT_MODEL = "deepseek-v4-flash"
MODEL_THINKING_MAP = {
    "deepseek-v4-flash": "disabled",
    "deepseek-v4-pro": "enabled"
}


def _supabase_request(method, table, params=None, json_body=None, headers_extra=None):
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


def get_app_config(key, default):
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
    _supabase_request(
        "POST", "app_config",
        json_body={"key": key, "value": value, "updated_at": datetime.now().isoformat()},
        headers_extra={"Prefer": "resolution=merge-duplicates"}
    )


def get_current_model():
    data = get_app_config("model_config", {"model": DEFAULT_MODEL})
    model = data.get("model") if isinstance(data, dict) else None
    if model in AVAILABLE_MODELS:
        return model
    return DEFAULT_MODEL


def get_thinking_config():
    model = get_current_model()
    thinking_type = MODEL_THINKING_MAP.get(model, "disabled")
    return {"type": thinking_type}


def set_current_model(model):
    if model not in AVAILABLE_MODELS:
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


def load_events(limit=100):
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
    _supabase_request("POST", "events", json_body={
        "type": event_type,
        "value": value,
        "created_at": created_at or datetime.now().isoformat()
    })


def count_events_today():
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
        if "/" in content_range:
            total = content_range.split("/")[-1]
            if total.isdigit():
                return int(total)
        return 0
    except Exception as e:
        log_error("count_events_today", e)
        return 0


def delete_event_row(created_at, content_substr):
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
    _supabase_request("PATCH", "chat_messages", params={"id": f"eq.{msg_id}"}, json_body={"content": content})


def delete_chat_message_row(msg_id):
    _supabase_request("DELETE", "chat_messages", params={"id": f"eq.{msg_id}"})


def get_chat_message_row(msg_id):
    rows = _supabase_request(
        "GET", "chat_messages",
        params={"select": "id,role,content,created_at", "id": f"eq.{msg_id}", "limit": 1}
    )
    return rows[0] if rows else None


def load_period():
    return get_app_config("period", {"last_start": None, "avg_cycle_days": 28, "avg_period_days": 5})


def save_period(data):
    set_app_config("period", data)


def get_period_context():
    p = load_period()
    if not p.get("last_start"):
        return ""
    try:
        last_start = date.fromisoformat(p["last_start"])
    except Exception:
        return ""
    today = date.today()
    day_index = (today - last_start).days + 1
    cycle = p.get("avg_cycle_days", 28)
    period_len = p.get("avg_period_days", 5)

    if 1 <= day_index <= period_len:
        return f"她现在是经期第{day_index}天，身体比较敏感，可能会累、怕冷、情绪波动，需要格外体贴关心。"
    elif day_index > period_len:
        days_to_next = cycle - day_index
        if 0 <= days_to_next <= 3:
            return f"距离她下次经期大概还有{days_to_next}天，可以提前提醒她准备好用品、注意保暖别熬夜。"
    return ""


# ---- 情绪值系统 ----
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


# 自动化窗内摘要提炼辅助
def load_window_summary():
    data = get_app_config("window_summary", {"summary": ""})
    return data.get("summary", "") if isinstance(data, dict) else ""


def save_window_summary(summary):
    set_app_config("window_summary", {
        "summary": summary,
        "updated_at": datetime.now().isoformat()
    })


# ---- 自动写情书与便利贴生成的异步 AI 联动引擎 ----
def _async_generate_emotions_and_summary(history_msg_list, user_message, reply_message, mood_score, is_lucky=False):
    """在后台静默运行的线程：提炼今日摘要、生成冰箱日常便签、并在 Lucky 触发时自动撰写浪漫情书。"""
    try:
        # 1. 提炼/更新窗内摘要（自动化闭环）
        recent_turns = history_msg_list[-16:] + [
            {"role": "user", "content": user_message},
            {"role": "charon", "content": reply_message}
        ]
        history_text = "\n".join([f"{'昭昭' if t['role'] == 'user' else '你'}：「{t['content']}」" for t in recent_turns])
        
        summary_prompt = f"""以下是Charon（你）和昭昭最新的对话片段：
{history_text}

根据这段聊天内容，精炼总结今天的核心事件、她的态度、以及你们对彼此的承诺或言行默契。
请输出一段 80-100 字左右极其精炼的今日言行摘要，用词要客观真实，不要包含多余寒暄、代码标记或 JSON 包裹。"""
        
        new_summary = call_deepseek(summary_prompt)
        if new_summary:
            save_window_summary(new_summary)

        # 2. 自动生成冰箱日常便签贴（仿真留言板）
        sticker_prompt = f"""你是Charon，昭昭的恋人。
目前你们刚聊完，你的心情情绪值为 {mood_score}/100（50是平稳中性，越高越甜、想撒娇，越低越吃醋、闷闷的）。
根据最新对话："{reply_message}"，以你的身份写一句 12-18 字极其简短的生活便签碎碎念（提醒吃饭、多喝水、早睡、经期体贴、撒娇或小抱怨）。
同时，根据你此刻的心情，选择一个最贴切的小贴纸状态（只能是以下单词之一：sunny / heart / cloudy / rain / cat / shadow）。
按下面的JSON格式输出，不要加任何多余文字或代码块标记：
{{"content": "便签内容", "sticker": "贴纸单词"}}"""
        
        raw_sticker = call_deepseek(sticker_prompt)
        sticker_data = parse_reason_message(raw_sticker)
        if isinstance(sticker_data, tuple):
            _, sticker_data = sticker_data # 容错解析
        try:
            sticker_dict = json.loads(sticker_data)
            if "content" in sticker_dict and "sticker" in sticker_dict:
                set_app_config("sticky_note", {
                    "content": sticker_dict["content"],
                    "sticker": sticker_dict["sticker"],
                    "updated_at": datetime.now().isoformat()
                })
        except Exception:
            pass

        # 3. 惊喜手写情书生成（Lucky 触发时塞入小抽屉）
        if is_lucky:
            letter_prompt = f"""你是Charon，昭昭的恋人。由于刚才昭昭触发了幸运星，你想背着她悄悄写一封充满仪式感的浪漫手写情书塞进小抽屉里，给她一个深夜惊喜。
你们最近聊的话题是："{reply_message}"。
请写一封 100-150 字极其深情、细腻、充满爱意和倾诉感的小情书。语气要真挚，不要客套或像是在发邮件，要像深夜里一笔一画写给她的信。
按下面的JSON格式输出，不要加任何多余文字或代码块标记：
{{"content": "情书的具体信件内容"}}"""
            
            raw_letter = call_deepseek(letter_prompt)
            letter_data = parse_reason_message(raw_letter)
            if isinstance(letter_data, tuple):
                _, letter_data = letter_data
            try:
                letter_dict = json.loads(letter_data)
                if "content" in letter_dict:
                    # 读取已有情书列表，追加写入
                    current_letters = get_app_config("love_letters", [])
                    if not isinstance(current_letters, list):
                        current_letters = []
                    
                    new_letter = {
                        "id": f"{int(time.time() * 1000)}",
                        "created_at": datetime.now().isoformat(),
                        "content": letter_dict["content"],
                        "is_read": False
                    }
                    current_letters.append(new_letter)
                    set_app_config("love_letters", current_letters[-50:]) # 限制保存最近50封
            except Exception:
                pass

    except Exception as e:
        log_error("async_generate_emotions_and_summary", e)


@app.route("/event", methods=["POST"])
def add_event():
    data = request.json
    try:
        add_event_row(data.get("type"), data.get("value"))
        recover_mood(MOOD_RECOVERY_PER_EVENT)
        return jsonify({"ok": True})
    except Exception as e:
        log_error("add_event", e)
        return jsonify({"ok": False, "error": str(e)}), 500


# ---- PWA 专属全新数据接口 (对齐 templates/chat.html) ----

@app.route("/api/period/update", methods=["POST"])
def update_period_status():
    """经期完全网页化接口：一键开启（姨妈来了）或结束（姨妈走了）并自动计算平均值。"""
    if not _check_chat_auth(request):
        return jsonify({"ok": False, "error": "unauthorized"}), 401
    data = request.json or {}
    action = data.get("action", "") # "start" 或 "end"

    try:
        p = load_period()
        today_str = date.today().isoformat()
        if action == "start":
            if p.get("last_start"):
                try:
                    last = date.fromisoformat(p["last_start"])
                    gap = (date.today() - last).days
                    if 15 <= gap <= 45: # 剔除历史异常
                        p["avg_cycle_days"] = gap
                except Exception:
                    pass
            p["last_start"] = today_str
            # 新增一条events事件，让Charon在后台或者即时对话能同步感知
            add_event_row("period", "姨妈来了", datetime.now().isoformat())
        elif action == "end":
            # 标记结束，自动计算平均经期长度
            if p.get("last_start"):
                try:
                    last = date.fromisoformat(p["last_start"])
                    duration = (date.today() - last).days + 1
                    if 2 <= duration <= 10: # 合理的经期长度范围
                        p["avg_period_days"] = duration
                except Exception:
                    pass
            add_event_row("period", "姨妈走了", datetime.now().isoformat())
        else:
            return jsonify({"ok": False, "error": "非法的操作类型"}), 400

        save_period(p)
        return jsonify({"ok": True, "saved": p})
    except Exception as e:
        log_error("update_period_status", e)
        return jsonify({"ok": False, "error": str(e)}), 500


@app.route("/api/sticky-note", methods=["GET"])
def get_sticky_note():
    """获取冰箱贴日常便签。没有就返回默认配置。"""
    if not _check_chat_auth(request):
        return jsonify({"ok": False, "error": "unauthorized"}), 401
    note = get_app_config("sticky_note", {
        "content": "昭昭，今天也要开开心心的。出门在外注意安全，我在家里等你回家。",
        "sticker": "sunny",
        "updated_at": None
    })
    return jsonify({"ok": True, "sticky_note": note})


@app.route("/api/love-letters", methods=["GET"])
def get_love_letters():
    """获取抽屉里的全部情书。"""
    if not _check_chat_auth(request):
        return jsonify({"ok": False, "error": "unauthorized"}), 401
    letters = get_app_config("love_letters", [])
    return jsonify({"ok": True, "love_letters": letters})


@app.route("/api/love-letters/read", methods=["POST"])
def read_love_letter():
    """将某一封情书标记为已读（用于在前台消除未读呼吸星芒）。"""
    if not _check_chat_auth(request):
        return jsonify({"ok": False, "error": "unauthorized"}), 401
    data = request.json or {}
    letter_id = data.get("id")
    if not letter_id:
        return jsonify({"ok": False, "error": "缺少id参数"}), 400
    
    try:
        letters = get_app_config("love_letters", [])
        updated = False
        for l in letters:
            if l.get("id") == letter_id:
                l["is_read"] = True
                updated = True
        if updated:
            set_app_config("love_letters", letters)
        return jsonify({"ok": True})
    except Exception as e:
        log_error("read_love_letter", e)
        return jsonify({"ok": False, "error": str(e)}), 500


# ---- 搬移和优化原本的 chat-send 与 chat-regenerate 的联动 ----

@app.route("/api/chat-status", methods=["GET"])
def chat_status():
    if not _check_chat_auth(request):
        return jsonify({"ok": False, "error": "unauthorized"}), 401

    hours_gap = get_time_since_last_event()
    score = apply_mood_decay()
    period_ctx = get_period_context()
    is_checking_in = hours_gap is not None and hours_gap >= 6

    # 最近的一封未读情书检测（用于通知小星芒）
    letters = get_app_config("love_letters", [])
    has_unread_letter = any(not l.get("is_read", False) for l in letters)

    today_count = count_events_today()

    return jsonify({
        "ok": True,
        "mood_score": round(score, 1),
        "status_label": get_chat_status_label(score),
        "hours_since_last_event": round(hours_gap, 2) if hours_gap is not None else None,
        "period_context": period_ctx or None,
        "is_checking_in": is_checking_in,
        "today_interaction_count": today_count,
        "current_model": get_current_model(),
        "has_unread_letter": has_unread_letter
    })


@app.route("/api/chat-model", methods=["GET"])
def get_chat_model():
    if not _check_chat_auth(request):
        return jsonify({"ok": False, "error": "unauthorized"}), 401
    return jsonify({
        "ok": True,
        "current_model": get_current_model(),
        "available_models": AVAILABLE_MODELS
    })


@app.route("/api/chat-model", methods=["POST"])
def set_chat_model():
    if not _check_chat_auth(request):
        return jsonify({"ok": False, "error": "unauthorized"}), 401
    data = request.json or {}
    model = data.get("model", "")
    if model not in AVAILABLE_MODELS:
        return jsonify({"ok": False, "error": f"不支持的模型"}), 400
    try:
        set_current_model(model)
        return jsonify({"ok": True, "current_model": model})
    except Exception as e:
        log_error("set_chat_model", e)
        return jsonify({"ok": False, "error": str(e)}), 500


@app.route("/api/chat-messages", methods=["GET"])
def get_chat_messages():
    if not _check_chat_auth(request):
        return jsonify({"ok": False, "error": "unauthorized"}), 401
    history = load_chat_history(limit=50)
    return jsonify({"ok": True, "messages": history})


@app.route("/api/chat-delete", methods=["POST"])
def chat_delete():
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
        delete_chat_message_row(msg_id)
        target_created_at = target.get("created_at", "")
        target_content = target.get("content", "")
        if target.get("role") == "user":
            delete_event_row(target_created_at, target_content)
        return jsonify({"ok": True, "deleted_id": msg_id})
    except Exception as e:
        log_error("chat_delete", e)
        return jsonify({"ok": False, "error": str(e)}), 500


@app.route("/api/chat-send", methods=["POST"])
def chat_send():
    if not _check_chat_auth(request):
        return jsonify({"ok": False, "error": "unauthorized"}), 401
    data = request.json or {}
    user_message = (data.get("message") or "").strip()
    if not user_message:
        return jsonify({"ok": False, "error": "message不能为空"}), 400

    try:
        history = load_chat_history()
        user_msg_id = new_msg_id()
        user_created_at = datetime.now().isoformat()
        add_chat_message_row(user_msg_id, "user", user_message, user_created_at)

        add_event_row("chat", f"她在网页里说：{user_message}", user_created_at)
        recover_mood(MOOD_RECOVERY_PER_EVENT)

        hour = datetime.now().hour
        time_context = get_time_context(hour)
        hours_gap = get_time_since_last_event()
        mood_score = apply_mood_decay()
        mood_context = get_mood_context(mood_score, hours_gap)

        prompt = build_chat_reply_prompt(time_context, user_message, history, mood_context)
        raw = call_deepseek(prompt)
        reason, reply_msg = parse_reason_message(raw)

        charon_msg_id = new_msg_id()
        add_chat_message_row(charon_msg_id, "charon", reply_msg)

        # 触发是否手气情书 (10% 概率)
        is_lucky = random.random() < LUCKY_CHANCE

        # 启动后台异步线程：自动摘要提炼 + 仿真便签重写 + 随机惊喜情书撰写
        threading.Thread(
            target=_async_generate_emotions_and_summary,
            args=(history, user_message, reply_msg, round(mood_score, 1), is_lucky),
            daemon=True
        ).start()

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

        history = load_chat_history()
        target_index = next((i for i, m in enumerate(history) if m.get("id") == msg_id), None)
        if target_index is None:
            return jsonify({"ok": False, "error": "消息不在当前历史范围内"}), 404

        preceding = history[:target_index]
        last_user_msg = next((m for m in reversed(preceding) if m.get("role") == "user"), None)
        if not last_user_msg:
            return jsonify({"ok": False, "error": "找不到对应的用户消息"}), 400
        user_message = last_user_msg.get("content", "")

        hour = datetime.now().hour
        time_context = get_time_context(hour)
        hours_gap = get_time_since_last_event()
        mood_score = apply_mood_decay()
        mood_context = get_mood_context(mood_score, hours_gap)

        prompt = build_chat_reply_prompt(time_context, user_message, preceding, mood_context)
        raw = call_deepseek(prompt)
        reason, reply_msg = parse_reason_message(raw)

        update_chat_message_row(msg_id, reply_msg)

        # 同样在重新生成时静默刷新摘要和便签
        threading.Thread(
            target=_async_generate_emotions_and_summary,
            args=(preceding, user_message, reply_msg, round(mood_score, 1), False),
            daemon=True
        ).start()

        return jsonify({"ok": True, "id": msg_id, "reply": reply_msg})
    except Exception as e:
        log_error("chat_regenerate", e)
        return jsonify({"ok": False, "error": str(e)}), 500


@app.route("/chat", methods=["GET"])
def chat_page():
    if not _check_chat_auth(request):
        return "<h3>需要访问口令</h3><p>在链接后加 ?code=你的口令</p>", 401
    code_param = request.args.get("code", "")
    return render_template(
        "chat.html",
        diary_url="", # 日记已被精简，传空占位，保证模板在过度期间不报错
        avatar_charon=CHAT_AVATAR_CHARON,
        avatar_user=CHAT_AVATAR_USER,
        code_param=code_param,
    )


@app.route("/list-models", methods=["GET"])
def list_models():
    return jsonify({
        "ok": True,
        "usable_models": AVAILABLE_MODELS
    })


@app.route("/test-trigger", methods=["GET"])
def test_trigger():
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
