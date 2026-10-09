"""Live desktop window for sign snapshots received by server.py."""
import json
from io import BytesIO
import tkinter as tk
from tkinter import ttk
from urllib.error import URLError
from urllib.request import urlopen
from PIL import Image, ImageTk


API_URL = "http://127.0.0.1:8765/api/state"
IMAGE_URL = "http://127.0.0.1:8765/api/image"
POLL_MS = 200


class SignMonitor:
    def __init__(self, root):
        self.root = root
        self.root.title("Schilder-Scanner · PC-Empfang")
        self.root.geometry("1040x680")
        self.root.minsize(780, 500)
        self.last_received = None
        self.last_signs_key = None
        self.preview_image = None

        frame = ttk.Frame(root, padding=14)
        frame.pack(fill="both", expand=True)
        ttk.Label(frame, text="Empfangene Schilder", font=("Segoe UI", 17, "bold")).pack(anchor="w")
        self.status = ttk.Label(frame, text="Verbinde mit Backend …")
        self.status.pack(anchor="w", pady=(4, 10))

        content = ttk.Panedwindow(frame, orient="horizontal")
        content.pack(fill="both", expand=True)
        left = ttk.Frame(content)
        right = ttk.Frame(content, padding=(12, 0, 0, 0))
        content.add(left, weight=3)
        content.add(right, weight=2)

        columns = ("sign", "value", "text", "confidence")
        self.table = ttk.Treeview(left, columns=columns, show="headings", height=8)
        for col, title, width in (
            ("sign", "Schild", 210), ("value", "Wert", 100),
            ("text", "Erkannter Text / Zustand", 280), ("confidence", "Formwert", 100),
        ):
            self.table.heading(col, text=title)
            self.table.column(col, width=width, anchor="w")
        self.table.pack(fill="x")

        ttk.Label(left, text="Eingangsprotokoll", font=("Segoe UI", 11, "bold")).pack(anchor="w", pady=(14, 5))
        log_frame = ttk.Frame(left)
        log_frame.pack(fill="both", expand=True)
        self.log = tk.Text(log_frame, height=10, wrap="word", state="disabled", font=("Consolas", 10))
        scrollbar = ttk.Scrollbar(log_frame, orient="vertical", command=self.log.yview)
        self.log.configure(yscrollcommand=scrollbar.set)
        self.log.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        ttk.Label(right, text="JPEG-Vorschau", font=("Segoe UI", 11, "bold")).pack(anchor="w", pady=(0, 5))
        self.preview = ttk.Label(right, text="Noch kein Bild empfangen", anchor="center", relief="groove")
        self.preview.pack(fill="both", expand=True)

        self.root.after(100, self.poll)

    def append_log(self, line):
        self.log.configure(state="normal")
        self.log.insert("end", line + "\n")
        self.log.see("end")
        self.log.configure(state="disabled")

    def poll(self):
        try:
            with urlopen(API_URL, timeout=1.5) as response:
                data = json.load(response)
            received = data.get("receivedAt")
            image_status = " · JPEG gespeichert" if data.get("imageAvailable") else ""
            self.status.configure(text="Backend verbunden · letzter Empfang: " + (received or "noch keiner") + image_status)
            if received and received != self.last_received:
                self.last_received = received
                if data.get("imageAvailable"):
                    with urlopen(IMAGE_URL, timeout=2) as image_response:
                        image = Image.open(BytesIO(image_response.read())).convert("RGB")
                    image.thumbnail((480, 500), Image.Resampling.LANCZOS)
                    self.preview_image = ImageTk.PhotoImage(image)
                    self.preview.configure(image=self.preview_image, text="")
                else:
                    self.preview.configure(image="", text="Noch kein JPEG empfangen")
                    self.preview_image = None
                signs = data.get("signs", [])
                signs_key = json.dumps(signs, sort_keys=True, ensure_ascii=False)
                if signs_key != self.last_signs_key:
                    self.last_signs_key = signs_key
                    self.table.delete(*self.table.get_children())
                    if not signs:
                        self.table.insert("", "end", values=("Kein Schild erkannt", "", "", ""))
                        self.append_log(f"{received}  ·  keine Schilder im aktuellen Bild")
                    else:
                        self.append_log(f"{received}  ·  {len(signs)} Schild(er)")
                        for sign in signs:
                            value = f"{sign['value']} km/h" if sign.get("value") is not None else ""
                            confidence = sign.get("confidence")
                            confidence_text = f"{round(confidence * 100)} %" if isinstance(confidence, (int, float)) else ""
                            self.table.insert("", "end", values=(sign.get("name", sign.get("label", "")), value, sign.get("text", ""), confidence_text))
                            self.append_log("    " + " · ".join(part for part in (sign.get("name", ""), value, sign.get("text", ""), confidence_text) if part))
        except (URLError, TimeoutError, OSError, json.JSONDecodeError, Image.UnidentifiedImageError):
            self.status.configure(text="Backend nicht erreichbar · zuerst python server.py starten")
        self.root.after(POLL_MS, self.poll)


if __name__ == "__main__":
    window = tk.Tk()
    SignMonitor(window)
    window.mainloop()
