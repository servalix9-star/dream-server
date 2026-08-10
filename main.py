import subprocess
subprocess.run(["pip", "install", "requests"], capture_output=True)

from flask import Flask, request, jsonify
from datetime import datetime
import json, os, requests, threading, time, traceback

app = Flask(__name__)
EVENTS_FILE = "events.json"
DIARY_FILE = "diary.json"
ERROR_LOG = "error.log"

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
BARK_KEY = os.environ.get("BARK_KEY")

# gemini-pro 和 gemini-2.0-flash 都已下线（2.0 Flash 于 2026-03-03 正式退役）
# 目前 free tier 可用的是 2.5 系列，flash-lite 配额更宽松，适合低频后台任务
GEMINI_MODEL = "gemini-2.5-flash-lite"


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
    """执行一次完整的：读取动态 -> 调 Gemini -> 发 Bark -> 写日记。
    单独抽出来，方便 keepalive 循环和手动测试接口共用同一份逻辑。"""
    if not GEMINI_API_KEY:
        raise RuntimeError("GEMINI_API_KEY not set")

    hour = datetime.now().hour
    events = load_events()
    if not events:
        recent = "最近没有任何活动记录"
    else:
        recent = "\n".join([f"{e['created_at'][:16]} {e['value']}" for e in events[-5:]])

    time_context = get_time_context(hour)
    prompt = build_prompt(time_context, recent)

    resp = requests.post(
        f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent?key={GEMINI_API_KEY}",
        json={"contents": [{"parts": [{"text": prompt}]}]},
        timeout=20
    )

    if resp.status_code != 200:
        raise RuntimeError(f"Gemini API error: status={resp.status_code} body={resp.text}")

    result = resp.json()

    if "candidates" not in result:
        raise RuntimeError(f"Gemini API unexpected response: {result}")

    msg = result["candidates"][0]["content"]["parts"][0]["text"].strip()

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
    """直接问 Gemini 这个 key 当前支持哪些模型，不用再猜名字。"""
    if not GEMINI_API_KEY:
        return jsonify({"ok": False, "error": "GEMINI_API_KEY not set"}), 500
    try:
        resp = requests.get(
            f"https://generativelanguage.googleapis.com/v1beta/models?key={GEMINI_API_KEY}",
            timeout=15
        )
        data = resp.json()
        # 只挑出支持 generateContent 的模型名，这才是能用来聊天的
        usable = []
        for m in data.get("models", []):
            methods = m.get("supportedGenerationMethods", [])
            if "generateContent" in methods:
                usable.append(m.get("name"))
        return jsonify({"ok": True, "usable_models": usable, "raw": data})
    except Exception as e:
        log_error("list_models", e)
        return jsonify({"ok": False, "error": str(e)}), 500


@app.route("/test-trigger", methods=["GET"])
def test_trigger():
    """手动触发一次，不用等55分钟。浏览器直接访问这个路径就行。"""
    try:
        msg = run_once()
        return jsonify({"ok": True, "msg": msg})
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
