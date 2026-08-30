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
    避免Charon下次醒来时recent里还看得到已经删掉的话。"""
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
    三栏布局：左侧波点纹理导航栏 + 中间深浅过渡Header对话区 + 右侧波点纹理状态面板（心里话斜体等样式）。"""
    if not _check_auth(request):
        return "<h3>需要访问口令</h3><p>在链接后加 ?code=你的口令</p>", 401

    code_param = request.args.get("code", "")
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
    /* 网格径向渐变，高亮度马卡龙粉 */
    background: radial-gradient(circle at 10% 20%, #fffcfd 0%, #fbf3f5 35%, #f3e2e6 70%, #ebd3d9 100%);
    background-attachment: fixed;
    font-family: "Songti SC", "STSong", Georgia, -apple-system, BlinkMacSystemFont, "PingFang SC", sans-serif;
    color: #5a4550;
  }}
  #app {{
    display: flex;
    height: 100vh; height: 100dvh;
    padding: 14px;
    gap: 14px;
    position: relative;
  }}

  /* ---- 左侧波点渐变导航栏 ---- */
  #nav {{
    width: 60px;
    flex-shrink: 0;
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 16px;
    padding: 24px 0;
    background: #f4e5eb; /* 粉色底色 */
    border-radius: 24px;
    position: relative;
    overflow: hidden;
    box-shadow: 0 8px 32px 0 rgba(184, 118, 138, 0.08);
  }}
  /* 纯 CSS 渐变波点掩膜层 (上至下：黑、粉、白、粉、黑) */
  #nav::before {{
    content: '';
    position: absolute;
    inset: 0;
    background: linear-gradient(to bottom, #1d1726 0%, #e599a9 30%, #ffffff 50%, #e599a9 70%, #1d1726 100%);
    -webkit-mask-image: radial-gradient(circle, #000 20%, transparent 22%), radial-gradient(circle, #000 20%, transparent 22%);
    -webkit-mask-size: 12px 12px;
    -webkit-mask-position: 0 0, 6px 6px;
    pointer-events: none;
    opacity: 0.8;
  }}
  .nav-icon {{
    width: 42px; height: 42px;
    border-radius: 14px;
    display: flex; align-items: center; justify-content: center;
    background: rgba(255, 255, 255, 0.45);
    color: #1d1726;
    font-size: 19px;
    text-decoration: none;
    border: 1px solid rgba(255, 255, 255, 0.6);
    transition: all 0.25s cubic-bezier(0.25, 0.8, 0.25, 1);
    z-index: 2; /* 确保在波点之上 */
  }}
  .nav-icon:hover {{
    background: rgba(255, 255, 255, 0.85);
    transform: translateY(-2px);
    color: #e599a9;
  }}
  .nav-icon:active {{ transform: translateY(0) scale(0.95); }}

  /* ---- 中间玻璃舱对话区 ---- */
  #main {{
    flex: 1;
    min-width: 0;
    display: flex;
    flex-direction: column;
    background: rgba(255, 255, 255, 0.16);
    backdrop-filter: blur(30px);
    -webkit-backdrop-filter: blur(30px);
    border-radius: 28px;
    box-shadow: 0 12px 40px rgba(184, 118, 138, 0.10);
    overflow: hidden;
    position: relative;
  }}
  
  /* 黑色渐变 Header：左侧深黑(含微波点) -> 右侧透明粉色 */
  #header {{
    padding: 20px 24px 18px;
    border-bottom: 1px solid rgba(255, 255, 255, 0.2);
    flex-shrink: 0;
    display: flex;
    align-items: center;
    justify-content: space-between;
    position: relative;
    background: 
      radial-gradient(rgba(255, 255, 255, 0.1) 15%, transparent 16%) 0 0/8px 8px,
      radial-gradient(rgba(255, 255, 255, 0.1) 15%, transparent 16%) 4px 4px/8px 8px,
      linear-gradient(90deg, #201724 0%, rgba(32, 23, 36, 0.9) 30%, rgba(255, 255, 255, 0) 100%);
  }}
  .header-left {{
    display: flex;
    align-items: center;
    gap: 14px;
  }}
  #header-avatar {{
    width: 44px; height: 44px;
    border-radius: 50%;
    object-fit: cover;
    flex-shrink: 0;
    border: 2px solid rgba(255, 255, 255, 0.8);
    box-shadow: 0 4px 12px rgba(184, 118, 138, 0.15);
  }}
  #header-text {{ min-width: 0; }}
  
  /* 做旧重影金属字 */
  #header .brand {{
    font-family: "Georgia", "Songti SC", serif;
    font-size: 26px;
    font-weight: 900;
    letter-spacing: 3px;
    background: linear-gradient(180deg, #ffffff 0%, #b09ba7 50%, #42303c 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    filter: drop-shadow(0px 2px 3px rgba(0, 0, 0, 0.45));
    line-height: 1.2;
    text-transform: uppercase;
  }}
  #header .sub {{
    font-size: 11px;
    color: #ebd1d7;
    letter-spacing: 1px;
    margin-top: 3px;
    display: flex;
    align-items: center;
    gap: 6px;
  }}
  /* 在线状态带上半透明白色背景胶囊框 */
  .online-tag {{
    display: inline-flex;
    align-items: center;
    gap: 5px;
    background: rgba(255, 255, 255, 0.15);
    border-radius: 20px;
    padding: 2px 10px;
    color: #ffffff;
    box-shadow: inset 0 1px 1px rgba(255, 255, 255, 0.1);
  }}
  .status-dot {{
    width: 6px; height: 6px;
    border-radius: 50%;
    background: #84cc9a;
    flex-shrink: 0;
    box-shadow: 0 0 8px #84cc9a;
  }}
  .status-dot.low {{ 
    background: #c3a1ad; 
    box-shadow: 0 0 8px #c3a1ad;
  }}
  
  /* 花体字手写签名 @Seraphina */
  .header-signature {{
    font-family: "Dancing Script", cursive;
    font-size: 20px;
    color: rgba(255, 255, 255, 0.85);
    text-shadow: 0 2px 4px rgba(0, 0, 0, 0.15);
    letter-spacing: 1px;
    margin-right: 10px;
  }}

  #messages {{
    flex: 1;
    overflow-y: auto;
    padding: 22px 24px;
    display: flex;
    flex-direction: column;
    gap: 16px;
    -webkit-overflow-scrolling: touch;
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
    border: 1.5px solid rgba(255, 255, 255, 0.85);
    box-shadow: 0 3px 8px rgba(184, 118, 138, 0.12);
  }}
  .msg-col {{
    display: flex;
    flex-direction: column;
    min-width: 0;
  }}
  .msg-row.user .msg-col {{ align-items: flex-end; }}
  .msg-row.charon .msg-col {{ align-items: flex-start; }}
  
  .bubble {{
    position: relative;
    padding: 12px 18px;
    line-height: 1.6;
    font-size: 15px;
    word-wrap: break-word;
    white-space: pre-wrap;
    cursor: pointer; /* 提示可以长按/右键 */
    user-select: none;
    -webkit-user-select: none;
  }}
  .bubble.user {{
    background: linear-gradient(135deg, #e599a9, #cf7d90);
    color: #ffffff;
    border-radius: 18px 4px 18px 18px; 
    box-shadow: 0 4px 15px rgba(184, 96, 118, 0.18);
  }}
  .bubble.charon {{
    background: rgba(255, 255, 255, 0.55);
    border: none;
    color: #6b5460;
    border-radius: 4px 18px 18px 18px; 
    box-shadow: 0 4px 15px rgba(184, 118, 138, 0.05);
  }}
  .bubble.pending {{ opacity: 0.5; }}
  
  /* 时间戳控制行（User侧移到气泡右下侧） */
  .msg-time-row {{
    display: flex;
    align-items: center;
    gap: 4px;
  }}
  .msg-time {{
    font-size: 10px;
    color: #b08d98;
    margin: 4px 4px 0;
  }}

  /* 网页底部输入栏铺满粉、白、黑波点渐变纹理 */
  #input-bar {{
    display: flex;
    gap: 10px;
    padding: 14px 20px calc(14px + env(safe-area-inset-bottom));
    border-top: 1px solid rgba(255, 255, 255, 0.2);
    flex-shrink: 0;
    background: #f4e5eb;
    position: relative;
  }}
  #input-bar::before {{
    content: '';
    position: absolute;
    inset: 0;
    background: linear-gradient(to right, #1d1726 0%, #e599a9 30%, #ffffff 50%, #e599a9 70%, #1d1726 100%);
    -webkit-mask-image: radial-gradient(circle, #000 15%, transparent 17%), radial-gradient(circle, #000 15%, transparent 17%);
    -webkit-mask-size: 10px 10px;
    -webkit-mask-position: 0 0, 5px 5px;
    pointer-events: none;
    opacity: 0.35;
  }}
  #input-bar textarea {{
    flex: 1;
    resize: none;
    border-radius: 16px;
    border: none;
    background: rgba(255, 255, 255, 0.7);
    color: #5a4550;
    padding: 11px 16px;
    font-size: 15px;
    font-family: inherit;
    max-height: 100px;
    outline: none;
    z-index: 2;
    transition: background 0.2s ease;
  }}
  #input-bar textarea:focus {{
    background: rgba(255, 255, 255, 0.95);
  }}
  #input-bar textarea::placeholder {{ color: #b08d98; }}
  #input-bar button {{
    border: none;
    border-radius: 16px;
    background: linear-gradient(135deg, #e599a9, #cf7d90);
    color: #fff;
    padding: 0 22px;
    font-size: 14px;
    cursor: pointer;
    box-shadow: 0 4px 12px rgba(184, 96, 118, 0.2);
    z-index: 2;
    transition: all 0.2s ease;
  }}
  #input-bar button:hover {{
    transform: translateY(-1px);
    box-shadow: 0 6px 16px rgba(184, 96, 118, 0.3);
  }}
  #input-bar button:active {{ transform: translateY(0); }}
  #input-bar button:disabled {{ opacity: 0.4; box-shadow: none; cursor: default; }}

  /* ---- 右侧状态面板（铺满粉、白渐变波点纹理） ---- */
  #panel {{
    width: 210px;
    flex-shrink: 0;
    display: none;
    flex-direction: column;
    gap: 16px;
    padding: 22px 18px;
    background: #f6eaee; 
    border-radius: 24px;
    box-shadow: 0 8px 32px rgba(184, 118, 138, 0.08);
    overflow-y: auto;
    position: relative;
  }}
  @media (min-width: 720px) {{
    #panel {{ display: flex; }}
  }}
  #panel::before {{
    content: '';
    position: absolute;
    inset: 0;
    background: linear-gradient(to bottom, #ffffff 0%, #e599a9 50%, #ffffff 100%);
    -webkit-mask-image: radial-gradient(circle, #000 16%, transparent 18%), radial-gradient(circle, #000 16%, transparent 18%);
    -webkit-mask-size: 12px 12px;
    -webkit-mask-position: 0 0, 6px 6px;
    pointer-events: none;
    opacity: 0.75;
  }}
  .panel-title {{
    font-family: "Georgia", "Songti SC", serif;
    font-size: 14px;
    letter-spacing: 3px;
    color: #a66275;
    margin-bottom: 6px;
    text-transform: uppercase;
    font-weight: bold;
    z-index: 2;
  }}
  .stat-block {{ margin-bottom: 12px; z-index: 2; }}
  .stat-label {{
    font-size: 11px;
    color: #b08d98;
    margin-bottom: 5px;
    display: flex;
    justify-content: space-between;
    align-items: baseline;
  }}
  
  /* 情绪值进度条：完美复刻参考图（2px 极简扁平细直轨样式，无圆角） */
  .stat-bar-track {{
    height: 2px; 
    border-radius: 0;
    background: rgba(90, 69, 80, 0.1); 
    overflow: hidden;
    width: 100%;
  }}
  .stat-bar-fill {{
    height: 100%;
    border-radius: 0;
    background: #cf7d90;
    transition: width 0.4s ease;
  }}
  .stat-value {{
    font-size: 11px;
    color: #b08d98;
  }}
  .period-tag {{
    font-size: 11px;
    color: #a66275;
    background: rgba(255, 255, 255, 0.5);
    border-radius: 10px;
    padding: 8px 10px;
    line-height: 1.5;
  }}
  .checking-tag {{
    font-size: 11px;
    color: #7d5a68;
    background: rgba(255, 255, 255, 0.5);
    border-radius: 10px;
    padding: 8px 10px;
    line-height: 1.5;
  }}
  .lucky-tag {{
    font-size: 11px;
    color: #b86076;
    background: linear-gradient(135deg, rgba(255, 255, 255, 0.6), rgba(255, 255, 255, 0.45));
    border-radius: 10px;
    padding: 8px 10px;
    line-height: 1.5;
  }}
  
  /* “心里话”卡片：完美恢复原版倾斜字体、颜色、气泡阴影与白透磨砂无框设计 */
  .thought-card {{
    font-size: 12px;
    color: #7a5a65; /* 彻底恢复原版深粉玫瑰灰字体颜色 */
    background: rgba(255, 255, 255, 0.6); 
    border-radius: 14px;
    padding: 12px 14px;
    line-height: 1.6;
    font-style: italic; /* 彻底恢复原版经典优雅倾斜体 */
    box-shadow: 0 4px 15px rgba(184, 118, 138, 0.05); /* 柔和阴影 */
  }}
  .summary-card {{
    font-size: 11px;
    color: #6e505f;
    background: rgba(255, 255, 255, 0.35);
    border-radius: 12px;
    padding: 9px 11px;
    line-height: 1.6;
    z-index: 2;
  }}
  .today-count {{
    font-size: 24px;
    color: #b86076;
    font-family: "Georgia", serif;
    font-weight: 600;
  }}
  .today-count-unit {{
    font-size: 11px;
    color: #b08d98;
    margin-left: 3px;
  }}
  .panel-empty {{
    font-size: 11px;
    color: #b08d98;
  }}

  /* ---- 微信式长按/右键撤回悬浮菜单 ---- */
  .context-menu {{
    position: fixed;
    background: rgba(29, 23, 38, 0.95);
    color: #e599a9;
    border-radius: 8px;
    padding: 8px 16px;
    font-size: 12px;
    font-weight: bold;
    cursor: pointer;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.25);
    z-index: 9999;
    display: none;
    transition: background 0.15s ease;
  }}
  .context-menu:hover {{
    background: #cf7d90;
    color: #ffffff;
  }}
</style>
</head>
<body>
<div id="app">

  <!-- 长按撤回悬浮菜单 -->
  <div id="msg-context-menu" class="context-menu">撤回</div>

  <div id="nav">
    <a class="nav-icon" href="/" title="首页">✦</a>
    <a class="nav-icon" href="{diary_url}" title="日记">✎</a>
  </div>

  <div id="main">
    <div id="header">
      <div class="header-left">
        <img id="header-avatar" src="{CHAT_AVATAR_CHARON}" alt="Charon">
        <div id="header-text">
          <div class="brand">CHARON</div>
          <div class="sub">
            <span class="online-tag">
              <span class="status-dot" id="status-dot"></span>
              <span id="status-label">加载中…</span>
            </span>
          </div>
        </div>
      </div>
      <!-- 右侧手写签名体 -->
      <div class="header-signature">@Seraphina</div>
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
const AVATAR_CHARON = {json.dumps(CHAT_AVATAR_CHARON)};
const AVATAR_USER = {json.dumps(CHAT_AVATAR_USER)};
const messagesEl = document.getElementById('messages');
const inputEl = document.getElementById('input');
const sendBtn = document.getElementById('send-btn');
const panelBody = document.getElementById('panel-body');
const statusDot = document.getElementById('status-dot');
const statusLabel = document.getElementById('status-label');
const contextMenu = document.getElementById('msg-context-menu');

let activeMsgId = null;
let activeRowEl = null;

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

  const timeRow = document.createElement('div');
  timeRow.className = 'msg-time-row';

  const timeEl = document.createElement('span');
  timeEl.className = 'msg-time';
  timeEl.textContent = formatTime(createdAt);
  timeRow.appendChild(timeEl);

  col.appendChild(timeRow);
  row.appendChild(col);

  // 绑定长按与右键菜单事件，专门处理撤回
  if (msgId) {{
    bindContextEvents(bubble, msgId, row);
  }}

  return row;
}}

/* 微信式长按/右键菜单绑定 */
function bindContextEvents(element, msgId, rowEl) {{
  let pressTimer;

  // 移动端长按
  element.addEventListener('touchstart', (e) => {{
    pressTimer = window.setTimeout(() => {{
      showMenu(e, msgId, rowEl);
    }}, 600);
  }});
  element.addEventListener('touchend', () => window.clearTimeout(pressTimer));
  element.addEventListener('touchmove', () => window.clearTimeout(pressTimer));

  // 网页端右键
  element.addEventListener('contextmenu', (e) => {{
    e.preventDefault();
    showMenu(e, msgId, rowEl);
  }});
}}

function showMenu(e, msgId, rowEl) {{
  activeMsgId = msgId;
  activeRowEl = rowEl;
  
  const x = e.clientX || (e.touches && e.touches[0].clientX);
  const y = e.clientY || (e.touches && e.touches[0].clientY);
  
  contextMenu.style.left = x + 'px';
  contextMenu.style.top = y + 'px';
  contextMenu.style.display = 'block';
}}

// 隐藏菜单
document.addEventListener('click', (e) => {{
  if (!e.target.classList.contains('context-menu')) {{
    contextMenu.style.display = 'none';
  }}
}});

// 触发菜单内撤回接口
contextMenu.addEventListener('click', async () => {{
  contextMenu.style.display = 'none';
  if (!activeMsgId || !activeRowEl) return;
  if (!confirm('撤回这条消息？')) return;
  
  try {{
    const res = await fetch(apiUrl('/api/chat-delete'), {{
      method: 'POST',
      headers: {{ 'Content-Type': 'application/json' }},
      body: JSON.stringify({{ id: activeMsgId }})
    }});
    const data = await res.json();
    if (data.ok) {{
      activeRowEl.remove();
    }} else {{
      alert('撤回失败：' + (data.error || '未知错误'));
    }}
  }} catch (e) {{
    alert('网络错误，没能撤回');
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
      messagesEl.innerHTML = '<div id="empty-hint">还没有聊过，说点什么吧</div>';
      return;
    }}
    data.messages.forEach(m => messagesEl.appendChild(renderMsgRow(m.role, m.content, m.created_at, false, m.id)));
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

function escapeHtml(str) {{
  const div = document.createElement('div');
  div.textContent = str;
  return div.innerHTML;
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

    // 更新header状态文字和状态点
    statusLabel.textContent = data.status_label || '在线';
    statusDot.className = 'status-dot' + (data.mood_score < 50 ? ' low' : '');

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
    html += '<div class="stat-block">';
    html += '<div class="stat-label"><span>今日互动</span></div>';
    html += '<div><span class="today-count">' + data.today_interaction_count + '</span><span class="today-count-unit">次</span></div>';
    html += '</div>';
    if (data.period_context) {{
      html += '<div class="stat-block"><div class="period-tag">' + escapeHtml(data.period_context) + '</div></div>';
    }}
    if (data.is_checking_in) {{
      html += '<div class="stat-block"><div class="checking-tag">好一阵没理TA了…</div></div>';
    }}
    if (data.last_was_lucky) {{
      html += '<div class="stat-block"><div class="lucky-tag">✨ 刚才是个惊喜消息</div></div>';
    }}
    if (data.last_thought) {{
      html += '<div class="stat-block"><div class="panel-title" style="margin-top:12px;">心里话</div><div class="thought-card">' + escapeHtml(data.last_thought) + '</div></div>';
    }}
    if (data.window_summary) {{
      html += '<div class="stat-block"><div class="panel-title" style="margin-top:12px;">最近聊过</div><div class="summary-card">' + escapeHtml(data.window_summary) + '</div></div>';
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
      
      // 成功获得消息ID，绑定长按撤回功能
      if (data.user_msg_id) {{
        userRow.dataset.msgId = data.user_msg_id;
        bindContextEvents(userRow.querySelector('.bubble'), data.user_msg_id, userRow);
      }}
      if (data.charon_msg_id) {{
        pendingRow.dataset.msgId = data.charon_msg_id;
        bindContextEvents(pendingBubble, data.charon_msg_id, pendingRow);
      }}
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