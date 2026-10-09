"""Serve the scanner website and receive its recognized-sign snapshots."""
from datetime import datetime, timezone
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import base64
import binascii
import json
from urllib.parse import urlparse


ROOT = Path(__file__).resolve().parent
latest_snapshot = {"receivedAt": None, "signs": [], "imageAvailable": False}
IMAGE_PATH = ROOT / "backend" / "latest-frame.jpg"


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
        global latest_snapshot
        if urlparse(self.path).path != "/api/signs":
            self.send_error(404, "Unknown API endpoint")
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0 or length > 1_000_000:
                raise ValueError("Invalid request size")
            data = json.loads(self.rfile.read(length))
            if not isinstance(data, dict):
                raise ValueError("Request must be a JSON object")
            signs = data.get("signs")
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
                    "text": str(sign.get("text", ""))[:300],
                    "confidence": sign.get("confidence") if isinstance(sign.get("confidence"), (int, float)) else None,
                })
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
                if len(image_bytes) > 700_000 or not image_bytes.startswith(b"\xff\xd8\xff"):
                    raise ValueError("JPEG image is too large or invalid")
            except (AttributeError, ValueError, binascii.Error) as error:
                self.send_error(400, f"Invalid JPEG: {error}")
                return
            IMAGE_PATH.parent.mkdir(parents=True, exist_ok=True)
            IMAGE_PATH.write_bytes(image_bytes)

        latest_snapshot = {
            "receivedAt": datetime.now(timezone.utc).isoformat(),
            "detectedAt": str(data.get("detectedAt", ""))[:60],
            "source": str(data.get("source", "website"))[:80],
            "signs": cleaned,
            "imageAvailable": image_bytes is not None or IMAGE_PATH.exists(),
        }
        payload = json.dumps({"ok": True, "receivedAt": latest_snapshot["receivedAt"]}).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


if __name__ == "__main__":
    address = ("0.0.0.0", 8765)
    print(f"Scanner-Website und Backend: http://localhost:{address[1]}/")
    print(f"Dashboard: http://localhost:{address[1]}/dashboard")
    print(f"API:       http://localhost:{address[1]}/api/state")
    ThreadingHTTPServer(address, Handler).serve_forever()
