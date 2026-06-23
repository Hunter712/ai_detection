import asyncio
import time
import logging
from contextlib import asynccontextmanager
import numpy as np
import cv2
import httpx
import uvicorn
from fastapi import FastAPI, BackgroundTasks
from picamera2 import Picamera2
from hailo_platform import (HEF, VDevice, HailoStreamInterface, ConfigureParams,
                            InputVStreamParams, OutputVStreamParams, FormatType, InferVStreams)

TELEGRAM_BOT_TOKEN = ""
TELEGRAM_CHAT_ID = ""
TIME_MASK = "%H%M%S_%d%m%Y"
MODEL_PATH = "/usr/local/hailo/resources/models/hailo8l/yolov8s.hef"

logging.basicConfig(
    filename='client.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt=TIME_MASK
)

picam2 = None
target_vdevice = None
net_group = None
input_vstreams_params = None
output_vstreams_params = None


@asynccontextmanager
async def lifespan():
    global picam2, target_vdevice, net_group, input_vstreams_params, output_vstreams_params

    logging.info("Initializing hardware resources...")
    picam2 = Picamera2()
    picam2.configure(picam2.create_video_configuration(main={"size": (640, 640), "format": "RGB888"}))
    picam2.start()

    hef = HEF(MODEL_PATH)
    target_vdevice = VDevice()
    target_vdevice.__enter__()

    net_group = target_vdevice.configure(hef, ConfigureParams.create_from_hef(hef, interface=HailoStreamInterface.PCIe))[0]
    input_vstreams_params = InputVStreamParams.make(net_group, format_type=FormatType.UINT8)
    output_vstreams_params = OutputVStreamParams.make(net_group, format_type=FormatType.FLOAT32)

    yield

    logging.info("Cleaning up resources...")
    if picam2:
        picam2.stop()
    if target_vdevice:
        target_vdevice.__exit__(None, None, None)
    logging.info("Resources successfully cleared.")


app = FastAPI(lifespan=lifespan)


async def send_photo_task(frame: np.ndarray, confidence: float):
    timestamp_file = time.strftime(TIME_MASK)
    conf_percent = f"{confidence * 100:.1f}"
    filename = f"person_{timestamp_file}_{conf_percent}%.jpg"

    success, encoded_image = cv2.imencode('.jpg', frame)
    if not success:
        logging.error("Failed to encode image in memory.")
        return

    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendPhoto"
    data = {
        "chat_id": TELEGRAM_CHAT_ID,
        "caption": f"🚨<b>Person detected!</b>\n<b>File:</b> {filename}",
        "parse_mode": "HTML"
    }
    files = {"photo": (filename, encoded_image.tobytes(), "image/jpeg")}

    async with httpx.AsyncClient(timeout=15.0) as client:
        try:
            response = await client.post(url, data=data, files=files)

            if response.status_code != 200:
                logging.error(f"[TG ERROR] Failed: {response.status_code} - {response.text}")

        except Exception as e:
            logging.error(f"[TG ERROR] Connection failed: {e}")


@app.post("/trigger")
async def trigger_motion(background_tasks: BackgroundTasks):  # Сделали функцию ASYNC
    global picam2, net_group, input_vstreams_params, output_vstreams_params

    best_confidence = 0.0
    best_frame = None
    frames_to_capture = 5

    with net_group.activate(net_group.create_params()), \
            InferVStreams(net_group, input_vstreams_params, output_vstreams_params) as infer_vstreams:

        for i in range(frames_to_capture):
            frame = picam2.capture_array()

            input_data = np.expand_dims(frame, axis=0).astype(np.uint8)
            infer_results = infer_vstreams.infer(input_data)

            person_detections = infer_results['yolov8s/yolov8_nms_postprocess'][0][0]
            confidence = float(np.max(person_detections[:, 4])) if person_detections.shape[0] > 0 else 0.0

            if confidence > best_confidence:
                best_confidence = confidence
                best_frame = frame.copy()

            if i < frames_to_capture - 1:
                await asyncio.sleep(0.05)

    if best_confidence > 0.5 and best_frame is not None:
        background_tasks.add_task(send_photo_task, best_frame, best_confidence)

    return {"status": "done", "max_confidence": best_confidence}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)