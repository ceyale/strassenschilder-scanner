"""Serve the scanner website and receive its recognized-sign snapshots."""
from datetime import datetime, timezone
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import base64
import binascii
import json
import math
import socketserver
import threading
import time
from urllib.parse import urlparse


ROOT = Path(__file__).resolve().parent
latest_snapshot = {"receivedAt": None, "signs": [], "imageAvailable": False}
IMAGE_PATH = ROOT / "backend" / "latest-frame.jpg"
COMMAND_PORT = 8766
COMMAND_INTERVAL = 0.1
CAMERA_TIMEOUT = 1.5
TRAFFIC_LIGHT_TIMEOUT = 1.0
TURN_SECONDS = 1.5
TURN_PEAK_DEG = 45
control_lock = threading.Lock()
control = {
    "drivePower": 30,
    "speedLimit": 30,
    "steeringAngle": 0,
    "trafficLight": "unbekannt",
    "stopReason": "",
    "stopRemainingMs": 0,
    "cameraConnected": False,
}
latest_snapshot["control"] = dict(control)
speed_limit = 30
traffic_light = "unbekannt"
traffic_light_seen_at = 0.0
stop_until = 0.0
turn_until = 0.0
turn_started = 0.0
turn_direction = ""
was_stopped_sign = False
was_turn_sign = ""


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(ROOT), **kwargs)

    def end_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        super().end_headers()

    def log_message(self, format, *args):
        # GUI/dashboard poll frequently; keep the server console readable.
        pass

    def do_OPTIONS(self):
        self.send_response(204)
        self.end_headers()

    def do_GET(self):
        if urlparse(self.path).path == "/api/state":
            payload = json.dumps(latest_snapshot, ensure_ascii=False).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
            return
        if urlparse(self.path).path == "/api/image":
            if not IMAGE_PATH.exists():
                self.send_error(404, "No JPEG frame has been received")
                return
            payload = IMAGE_PATH.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "image/jpeg")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
            return
        if urlparse(self.path).path == "/dashboard":
            self.path = "/backend/dashboard.html"
        super().do_GET()

    def do_POST(self):
        global latest_snapshot, speed_limit, traffic_light, traffic_light_seen_at, stop_until
        global turn_until, turn_started, turn_direction, was_stopped_sign, was_turn_sign
        endpoint = urlparse(self.path).path
        pc_frame_mode = endpoint == "/api/frame"
        if endpoint not in ("/api/signs", "/api/frame"):
            self.send_error(404, "Unknown API endpoint")
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0 or length > 2_600_000:
                raise ValueError("Invalid request size")
            data = json.loads(self.rfile.read(length))
            if not isinstance(data, dict):
                raise ValueError("Request must be a JSON object")
        except (ValueError, json.JSONDecodeError) as error:
            self.send_error(400, str(error))
            return

        image_data = data.get("imageJpeg")
        image_bytes = None
        if image_data is not None:
            try:
                header, encoded = image_data.split(",", 1)
                if header != "data:image/jpeg;base64":
                    raise ValueError("imageJpeg must be a JPEG data URL")
                image_bytes = base64.b64decode(encoded, validate=True)
                if len(image_bytes) > 1_800_000 or not image_bytes.startswith(b"\xff\xd8\xff"):
                    raise ValueError("JPEG image is too large or invalid")
            except (AttributeError, ValueError, binascii.Error) as error:
                self.send_error(400, f"Invalid JPEG: {error}")
                return

        vision_result = None
        if pc_frame_mode:
            if image_bytes is None:
                self.send_error(400, "PC mode requires imageJpeg")
                return
            try:
                from backend_vision import get_pc_vision
                vision_result = get_pc_vision().infer(image_bytes)
                signs = vision_result["detections"]
            except Exception as error:
                payload = json.dumps({"ok": False, "error": str(error)}, ensure_ascii=False).encode("utf-8")
                self.send_response(503)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)
                return
        else:
            signs = data.get("signs")

        try:
            if not isinstance(signs, list) or len(signs) > 50:
                raise ValueError("signs must be a list with at most 50 entries")
            cleaned = []
            for sign in signs:
                if not isinstance(sign, dict) or not isinstance(sign.get("label"), str):
                    raise ValueError("Each sign needs a label")
                cleaned.append({
                    "label": sign["label"][:80],
                    "name": str(sign.get("name", ""))[:120],
                    "value": sign.get("value") if isinstance(sign.get("value"), (int, float)) else None,
                    "state": str(sign.get("state", ""))[:40],
                    "direction": str(sign.get("direction", ""))[:20],
                    "text": str(sign.get("text", ""))[:300],
                    "confidence": sign.get("confidence") if isinstance(sign.get("confidence"), (int, float)) else None,
                })
                for coordinate in ("x", "y", "w", "h"):
                    if isinstance(sign.get(coordinate), (int, float)):
                        cleaned[-1][coordinate] = sign[coordinate]
        except (ValueError, json.JSONDecodeError) as error:
            self.send_error(400, str(error))
            return
        if image_bytes is not None:
            IMAGE_PATH.parent.mkdir(parents=True, exist_ok=True)
            IMAGE_PATH.write_bytes(image_bytes)

        now = time.monotonic()
        stop_seen = any(sign["label"] == "stop" for sign in cleaned)
        if stop_seen and not was_stopped_sign:
            stop_until = now + 5.0
        was_stopped_sign = stop_seen

        # Keep the speed limit from the strongest current speed-sign detection.
        # Snapshot ordering is not meaningful: a lower-confidence duplicate must
        # never overwrite the best candidate just because it appears later.
        speed_signs = [
            sign for sign in cleaned
            if sign["label"] == "verbot" and sign["value"] is not None
        ]
        if speed_signs:
            best_speed_sign = max(
                speed_signs,
                key=lambda sign: sign["confidence"] if sign["confidence"] is not None else -1,
            )
            # Speed-limit signs set motor power as a percentage, capped at 100%.
            speed_limit = max(0, min(100, int(best_speed_sign["value"])))

        light_candidates = []
        for sign in cleaned:
            if sign["label"] != "ampel":
                continue
            state = (sign["state"] or sign["text"]).strip().lower()
            state = {"grün": "gruen", "green": "gruen", "yellow": "gelb", "red": "rot"}.get(state, state)
            if state in ("rot", "gelb", "gruen"):
                confidence = sign["confidence"] if sign["confidence"] is not None else -1
                light_candidates.append((confidence, state))
        # Select the strongest visible light and retain it briefly between
        # camera frames. Once unseen for one second, treat it as absent.
        if light_candidates:
            traffic_light = max(light_candidates, key=lambda item: item[0])[1]
            traffic_light_seen_at = now
        elif traffic_light_seen_at and now - traffic_light_seen_at > TRAFFIC_LIGHT_TIMEOUT:
            # Only clear after a fresh camera snapshot confirms that no light
            # is visible. If the camera stops sending frames, CAMERA_TIMEOUT
            # will stop the truck instead of releasing a stale red/yellow light.
            traffic_light = "unbekannt"

        directions = [sign["direction"].lower() for sign in cleaned
                      if sign["label"] == "gebot" and sign["direction"].lower() in ("links", "rechts")]
        current_turn = directions[0] if directions else ""
        if current_turn and current_turn != was_turn_sign:
            turn_direction = current_turn
            turn_started = now
            turn_until = now + TURN_SECONDS
        was_turn_sign = current_turn

        latest_snapshot = {
            "receivedAt": datetime.now(timezone.utc).isoformat(),
            "detectedAt": str(data.get("detectedAt", ""))[:60],
            "source": str(data.get("source", "website"))[:80],
            "signs": cleaned,
            "recognitionMode": "pc" if pc_frame_mode else "phone",
            "imageAvailable": image_bytes is not None or IMAGE_PATH.exists(),
        }
        refresh_control(now)
        response_data = {"ok": True, "receivedAt": latest_snapshot["receivedAt"]}
        if vision_result is not None:
            response_data.update({
                "detections": cleaned,
                "imageWidth": vision_result["imageWidth"],
                "imageHeight": vision_result["imageHeight"],
                "inferenceMs": vision_result["inferenceMs"],
                "model": vision_result["model"],
            })
        payload = json.dumps(response_data, ensure_ascii=False).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


def refresh_control(now=None):
    """Translate the latest phone snapshot into a small EV3 motor command."""
    if now is None:
        now = time.monotonic()
    received = latest_snapshot.get("receivedAt")
    camera_connected = False
    if received:
        try:
            seen_at = datetime.fromisoformat(received).timestamp()
            camera_connected = time.time() - seen_at <= CAMERA_TIMEOUT
        except (TypeError, ValueError):
            pass
    stop_reason = ""
    if not camera_connected:
        stop_reason = "Kamera-/Backend-Verbindung unterbrochen"
    elif traffic_light in ("rot", "gelb"):
        stop_reason = "Ampel " + traffic_light
    elif now < stop_until:
        stop_reason = "Stoppschild"

    steering_angle = 0
    if now < turn_until and turn_until > turn_started:
        progress = (now - turn_started) / TURN_SECONDS
        # Smooth 0 → full steering → 0 with zero velocity at both ends.
        easing = (1 - math.cos(2 * math.pi * progress)) / 2
        steering_angle = round(TURN_PEAK_DEG * easing) * (1 if turn_direction == "links" else -1)

    with control_lock:
        control.update({
            "drivePower": 0 if stop_reason else speed_limit,
            "speedLimit": speed_limit,
            "steeringAngle": steering_angle,
            "trafficLight": traffic_light,
            "stopReason": stop_reason,
            "stopRemainingMs": max(0, int((stop_until - now) * 1000)),
            "cameraConnected": camera_connected,
        })
        latest_snapshot["control"] = dict(control)


class EV3CommandHandler(socketserver.BaseRequestHandler):
    def handle(self):
        # EV3 connects to this laptop TCP port and receives newline-delimited JSON.
        while True:
            try:
                refresh_control()
                with control_lock:
                    message = json.dumps(control, ensure_ascii=False, separators=(",", ":")) + "\n"
                self.request.sendall(message.encode("utf-8"))
                time.sleep(COMMAND_INTERVAL)
            except (OSError, ConnectionError):
                break


class ThreadedCommandServer(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True


if __name__ == "__main__":
    address = ("0.0.0.0", 8765)
    command_server = ThreadedCommandServer(("0.0.0.0", COMMAND_PORT), EV3CommandHandler)
    threading.Thread(target=command_server.serve_forever, daemon=True).start()
    print(f"Scanner-Website und Backend: http://localhost:{address[1]}/")
    print(f"Dashboard: http://localhost:{address[1]}/dashboard")
    print(f"API:       http://localhost:{address[1]}/api/state")
    print(f"Bluetooth-Gamepad-Brücke (WSL) liest Befehle von TCP-Port {COMMAND_PORT}")
    ThreadingHTTPServer(address, Handler).serve_forever()
