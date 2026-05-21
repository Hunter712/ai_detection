import os
import time
import threading
import requests
import gi

os.environ["GST_PLUGIN_FEATURE_RANK"] = "vaapidecodebin:NONE"
gi.require_version("Gst", "1.0")

import hailo
from hailo_apps.python.pipeline_apps.detection.detection_pipeline import GStreamerDetectionApp
from hailo_apps.python.core.gstreamer.gstreamer_app import app_callback_class

SERVER_URL = ""
COOLDOWN_SECONDS = 10


class UserData(app_callback_class):
    def __init__(self):
        super().__init__()
        self.last_send_time = 0


def send_webhook(confidence):
    payload = {
        "event": "person_detected",
        "confidence": round(confidence, 2),
        "timestamp": str(int(time.time()))
    }
    try:
        r = requests.post(SERVER_URL, json=payload, timeout=5)
        print(f"{payload}  {r.status_code}")
    except Exception as e:
        print(f"error e")

def app_callback(element, buffer, user_data):
    if buffer is None:
        return

    detections = hailo.get_roi_from_buffer(buffer).get_objects_typed(hailo.HAILO_DETECTION)
    person_detected = False
    max_confidence = 0.0

    for detection in detections:
        if detection.get_label() == "person" and detection.get_confidence() > 0.5:
            person_detected = True
            if detection.get_confidence() > max_confidence:
                max_confidence = detection.get_confidence()

    if person_detected:
        current_time = time.time()
        if current_time - user_data.last_send_time > COOLDOWN_SECONDS:
            user_data.last_send_time = current_time
            threading.Thread(target=send_webhook, args=(max_confidence,), daemon=True).start()


def main():
    app = GStreamerDetectionApp(app_callback, UserData())
    app.run()


if __name__ == "__main__":
    main()
