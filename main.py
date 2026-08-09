from flask import Flask, request, jsonify
from datetime import datetime
import json, os

app = Flask(__name__)
EVENTS_FILE = "events.json"

def load_events():
    if os.path.exists(EVENTS_FILE):
        with open(EVENTS_FILE, "r") as f:
            return json.load(f)
    return []

def save_events(events):
    with open(EVENTS_FILE, "w") as f:
        json.dump(events, f)

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

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
