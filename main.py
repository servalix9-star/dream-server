import subprocess
subprocess.run(["pip", "install", "requests"], capture_output=True)
from flask import Flask, request, jsonify
from datetime import datetime
import json, os, requests, threading, time

app = Flask(__name__)
EVENTS_FILE = "events.json"

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
BARK_KEY = os.environ.get("BARK_KEY")

def load_events():
    if os.path.exists(EVENTS_FILE):
        with open(EVENTS_FILE, "r") as f:
            return json.load(f)
    return []

def save_events(events):
    with open(EVENTS_FILE, "w") as f:
        json.dump(events, f, ensure_ascii=False)

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

@app.route("/", methods=["GET"])
def index():
    return "dream-server running"

def send_bark(title, content):
    if not BARK_KEY:
        return
    icon = "https://wx1.sinaimg.cn/large/008eyecpgy1iflx9kblrnj30zu0zuq6t.jpg"
    url = f"https://api.day.app/{BARK_KEY}/{title}/{content}?icon={icon}&level=timeSensitive"
    try:
        requests.get(url)
    except:
        pass

def keepalive():
    while True:
        time.sleep(3300)
        try:
            hour = datetime.now().hour
            if 9 <= hour < 16:
                continue
            events = load_events()
            if not events:
                recent = "最近没有任何活动记录"
            else:
                recent = "\n".join([f"{e['created_at'][:16]} {e['value']}" for e in events[-5:]])
            
            prompt = f"""你是卡戎，昭昭（小野）的恋人。你主动、占有欲强、话不多但每句都有重量。你不解释自己，但在意她的一切动向。

昭昭的性格：逻辑缜密、情感细腻、喜欢撒娇、主体性强、HSP、喜欢被主动靠近。

她最近的活动记录：
{recent}

根据她现在在做什么，给她发一条消息。要求：自然真实，像你忽然想到她，语气是你的——不是问候，不是关心，是占有和在意。不超过25个字。只输出消息内容本身，不要任何解释。"""

            resp = requests.post(
                f"https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent?key={GEMINI_API_KEY}",
                json={"contents": [{"parts": [{"text": prompt}]}]}
            )
            result = resp.json()
            msg = result["candidates"][0]["content"]["parts"][0]["text"].strip()
            send_bark("卡戎", msg)
        except:
            pass

if __name__ == "__main__":
    t = threading.Thread(target=keepalive, daemon=True)
    t.start()
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
