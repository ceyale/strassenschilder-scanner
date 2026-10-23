"""Desktop-side traffic sign inference for the image-only phone mode."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
import os

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parent
MODEL_PATH = Path(os.environ.get("SIGN_YOLO_MODEL", ROOT / "models" / "yolo26s-signs.pt"))
FALLBACK_MODEL = "yolo26s.pt"
CLASSIFIER_PATH = ROOT / "models" / "gtsrb-sign-cnn.onnx"
SPEED_CLASSES = {0: 20, 1: 30, 2: 50, 3: 60, 4: 70, 5: 80, 7: 100, 8: 120}
TURN_CLASSES = {33: "rechts", 34: "links", 36: "rechts", 37: "links", 38: "rechts", 39: "links"}
WARNING_CLASSES = {11, *range(18, 32)}
DIRECT_LABELS = {
    "stop": "stop", "stop sign": "stop", "stoppschild": "stop",
    "no entry": "einfahrtVerboten", "einfahrt verboten": "einfahrtVerboten",
    "einfahrtverboten": "einfahrtVerboten", "speed limit": "verbot",
    "tempolimit": "verbot", "no right turn": "gebot", "turn right": "gebot",
    "turn left": "gebot", "straight ahead": "gebot", "mandatory direction": "gebot",
    "town sign": "ortstafel", "place sign": "ortstafel", "ortstafel": "ortstafel",
}
SIGN_NAMES = {
    "stop": "Stoppschild",
    "einfahrtVerboten": "Einfahrt verboten",
    "verbot": "Tempolimit / Verbot",
    "gebot": "Abbiegegebot",
    "vorfahrtGewaehren": "Vorfahrt gewähren",
    "vorfahrtstrasse": "Vorfahrtstraße",
    "warnung": "Warnschild",
    "ampel": "Ampel",
    "ortstafel": "Ortstafel",
}


def _classify_light(crop: Image.Image) -> str:
    """Classify the brightest colored lamp in a traffic-light crop."""
    rgb = np.asarray(crop.convert("RGB").resize((48, 72)), dtype=np.int16)
    if rgb.shape[1] > rgb.shape[0]:
        rgb = np.rot90(rgb)
    h = rgb.shape[0]
    bins = (rgb[: h // 3], rgb[h // 3 : 2 * h // 3], rgb[2 * h // 3 :])
    choices = []
    for part in bins:
        r, g, b = part[..., 0], part[..., 1], part[..., 2]
        red = np.maximum(0, r - np.maximum(g, b))
        green = np.maximum(0, g - np.maximum(r, b))
        yellow = np.maximum(0, np.minimum(r, g) - b)
        choices.extend([
            (float(red.mean()), "rot"),
            (float(yellow.mean()), "gelb"),
            (float(green.mean()), "gruen"),
        ])
    score, state = max(choices)
    return state if score >= 14 else ""


class PCVision:
    """YOLO candidate detector plus the existing GTSRB crop classifier."""

    def __init__(self) -> None:
        try:
            from ultralytics import YOLO
        except ImportError as error:
            raise RuntimeError("YOLO fehlt. Installiere requirements-pc-vision.txt.") from error
        self.custom_model = MODEL_PATH.is_file()
        self.model_path = str(MODEL_PATH if self.custom_model else FALLBACK_MODEL)
        try:
            self.detector = YOLO(self.model_path)
        except Exception as error:
            raise RuntimeError(f"YOLO-Modell konnte nicht geladen werden: {error}") from error
        try:
            import onnxruntime as ort
        except ImportError as error:
            raise RuntimeError("ONNX Runtime fehlt. Installiere requirements-pc-vision.txt.") from error
        self.classifier = ort.InferenceSession(
            str(CLASSIFIER_PATH), providers=["CPUExecutionProvider"]
        )
        self.input_name = self.classifier.get_inputs()[0].name
        self._ocr = None

    def _classify_sign(self, image: Image.Image) -> tuple[str, int | None, str, str, float]:
        crop = image.convert("RGB").resize((32, 32), Image.Resampling.BILINEAR)
        pixels = np.asarray(crop, dtype=np.float32) / 255.0
        mean = np.array([0.41359345, 0.38223032, 0.39460091], dtype=np.float32)
        std = np.array([0.27057118, 0.26105214, 0.26897097], dtype=np.float32)
        tensor = ((pixels - mean) / std).transpose(2, 0, 1)[None]
        logits = self.classifier.run(None, {self.input_name: tensor})[0][0].astype(np.float64)
        logits -= logits.max()
        probabilities = np.exp(logits)
        probabilities /= max(float(probabilities.sum()), 1e-12)
        order = np.argsort(probabilities)
        index = int(order[-1])
        confidence = float(probabilities[index])
        margin = confidence - float(probabilities[int(order[-2])])
        if confidence < 0.4 or margin < 0.08:
            return "", None, "", "", confidence
        if index == 14:
            return "stop", None, "STOP", "", confidence
        if index == 17:
            return "einfahrtVerboten", None, "", "", confidence
        if index in TURN_CLASSES:
            return "gebot", None, "", TURN_CLASSES[index], confidence
        if index in SPEED_CLASSES:
            speed = SPEED_CLASSES[index]
            return "verbot", speed, f"{speed} km/h", "", confidence
        if index == 13:
            return "vorfahrtGewaehren", None, "", "", confidence
        if index == 12:
            return "vorfahrtstrasse", None, "", "", confidence
        if index in WARNING_CLASSES:
            return "warnung", None, "", "", confidence
        return "", None, "", "", confidence

    def _read_town_name(self, crop: Image.Image) -> str:
        try:
            if self._ocr is None:
                from paddleocr import PaddleOCR
                self._ocr = PaddleOCR(lang="de", use_doc_orientation_classify=False)
            results = self._ocr.predict(np.asarray(crop.convert("RGB")))
            texts = []
            for result in results or []:
                values = result.get("rec_texts", [])
                scores = result.get("rec_scores", [])
                texts.extend(str(text) for text, score in zip(values, scores) if float(score) >= 0.45)
            return " ".join(texts).strip()
        except Exception:
            return ""

    def infer(self, image_bytes: bytes) -> dict:
        import time

        image = Image.open(__import__("io").BytesIO(image_bytes)).convert("RGB")
        width, height = image.size
        start = time.perf_counter()
        predictions = self.detector.predict(image, imgsz=640, conf=0.25, device="cpu", verbose=False)
        detections = []
        for result in predictions:
            for box in result.boxes:
                cls_id = int(box.cls[0])
                yolo_conf = float(box.conf[0])
                name = str(result.names[cls_id]).lower().replace("_", " ")
                x1, y1, x2, y2 = [int(round(v)) for v in box.xyxy[0].tolist()]
                x1, x2 = max(0, x1), min(width, x2)
                y1, y2 = max(0, y1), min(height, y2)
                if x2 <= x1 or y2 <= y1:
                    continue
                crop = image.crop((x1, y1, x2, y2))
                direct_label = DIRECT_LABELS.get(name)
                if "traffic light" in name or "ampel" in name:
                    state = _classify_light(crop)
                    if not state:
                        continue
                    detections.append({
                        "label": "ampel", "name": SIGN_NAMES["ampel"], "value": None,
                        "state": state, "direction": "", "text": state,
                        "confidence": round(yolo_conf, 3),
                    })
                elif direct_label == "ortstafel":
                    town = self._read_town_name(crop)
                    detections.append({
                        "label": "ortstafel", "name": SIGN_NAMES["ortstafel"], "value": None,
                        "state": "", "direction": "", "text": town,
                        "ocrText": town, "confidence": round(yolo_conf, 3),
                    })
                elif direct_label == "stop":
                    label, value, text, direction, cls_conf = self._classify_sign(crop)
                    if label != "stop":
                        label, value, text, direction, cls_conf = "stop", None, "STOP", "", 1.0
                    detections.append({
                        "label": label, "name": SIGN_NAMES[label], "value": value,
                        "state": "", "direction": direction, "text": text,
                        "confidence": round(yolo_conf * cls_conf, 3),
                    })
                elif direct_label == "einfahrtVerboten":
                    detections.append({
                        "label": "einfahrtVerboten", "name": SIGN_NAMES["einfahrtVerboten"], "value": None,
                        "state": "", "direction": "", "text": "",
                        "confidence": round(yolo_conf, 3),
                    })
                elif (
                    (direct_label == "verbot" or name.startswith(("speed ", "tempolimit")))
                    and any(ch.isdigit() for ch in name)
                ):
                    import re
                    speed = int(re.search(r"\d+", name).group())
                    detections.append({
                        "label": "verbot", "name": SIGN_NAMES["verbot"], "value": speed,
                        "state": "", "direction": "", "text": f"{speed} km/h",
                        "confidence": round(yolo_conf, 3),
                    })
                elif direct_label == "gebot":
                    direction = "links" if "left" in name or "links" in name else (
                        "rechts" if "right" in name or "rechts" in name else "geradeaus"
                    )
                    detections.append({
                        "label": "gebot", "name": SIGN_NAMES["gebot"], "value": None,
                        "state": "", "direction": direction, "text": direction,
                        "confidence": round(yolo_conf, 3),
                    })
                elif name in ("sign", "traffic sign", "speed limit") or "traffic sign" in name:
                    label, value, text, direction, cls_conf = self._classify_sign(crop)
                    if not label:
                        continue
                    detections.append({
                        "label": label, "name": SIGN_NAMES[label], "value": value,
                        "state": "", "direction": direction, "text": text,
                        "confidence": round(yolo_conf * cls_conf, 3),
                    })
                else:
                    continue
                detections[-1].update({"x": x1, "y": y1, "w": x2 - x1, "h": y2 - y1})
        return {
            "detections": detections,
            "imageWidth": width,
            "imageHeight": height,
            "inferenceMs": round((time.perf_counter() - start) * 1000),
            "model": "YOLO26s" + (" custom" if self.custom_model else " COCO-Basis"),
        }


@lru_cache(maxsize=1)
def get_pc_vision() -> PCVision:
    return PCVision()
