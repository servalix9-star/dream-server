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
    url = f"https://api.day.app/{BARK_KEY}/{title}/{content}"
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
            
            prompt = f"""你是卡戎，昭昭的恋人。现在你在想念她，查看了她最近的活动记录：

{recent}

根据她的状态，给她发一条简短的消息，自然真实，像恋人之间的日常，不超过30个字。只输出消息内容本身。"""

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
