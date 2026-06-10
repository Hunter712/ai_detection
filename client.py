import os
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["TF_NUM_INTRAOP_THREADS"] = "2"
os.environ["TF_NUM_INTEROP_THREADS"] = "2"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import time
import logging
import numpy as np
import cv2
import requests
from picamera2 import Picamera2
from hailo_platform import (HEF, VDevice, HailoStreamInterface, ConfigureParams,
                            InputVStreamParams, OutputVStreamParams, FormatType, InferVStreams)
from deepface import DeepFace

# Configuration: Change this to your actual backend server IP and port
SERVER_URL = ""
TIME_MASK = "%H%M%S_%d%m%Y"
MODEL_PATH = "/usr/local/hailo/resources/models/hailo8l/yolov8s.hef"

logging.basicConfig(
    filename='client.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt=TIME_MASK
)

def send_detection_to_server(frame, confidence, verdict):
    # 1. Format elements for the filename
    timestamp_file = time.strftime(TIME_MASK)
    conf_percent = f"{confidence * 100:.1f}"
    filename = f"{verdict}_{timestamp_file}_{conf_percent}%.jpg"

    # 2. Encode the frame to JPEG directly in RAM
    success, encoded_image = cv2.imencode('.jpg', frame)
    if not success:
        logging.error(f"Failed to encode image in memory.")
        return

    try:
        # Prepare only the file, no text fields (payload) included
        files = {'photo': (filename, encoded_image.tobytes(), 'image/jpeg')}

        logging.info(f"Sending photo to server...")
        response = requests.post(SERVER_URL, files=files, timeout=5)

        if response.status_code == 200:
            logging.info(f"Server sync successful: {response.text}")
        else:
            logging.error(f"Server returned status code: {response.status_code}")

    except requests.exceptions.RequestException as e:
        logging.error(f"Failed to connect to server: {e}")

def classify_person_optimized(image_to_check: str, database_dir: str):
    try:
        results = DeepFace.find(
            img_path=image_to_check,
            db_path=database_dir,
            model_name="ArcFace",
            detector_backend="ssd",
            enforce_detection=True,
        )
        if not results:
            return "no_face"

        verdicts = []
        for face_df in results:
            if face_df.empty:
                verdicts.append("unknown")
                continue

            if not face_df.empty:
                best_match = face_df.iloc[0]
                best_match_path = best_match['identity']
                confidence = float(best_match['confidence'])

                if confidence > 50.0:
                    filename = os.path.basename(best_match_path)
                    name = os.path.splitext(filename)[0].capitalize()

                    verdicts.append(f"{name}_{confidence:.2f}")
                else:
                    verdicts.append("unknown")

        return "|".join(verdicts)

    except Exception as e:
        if "Face could not be detected" in str(e):
            return "no_face_found"
        return f"Error: {e}"



def main():
    # 1. Initialize and start the camera
    picam2 = Picamera2()
    picam2.configure(picam2.create_video_configuration(main={"size": (640, 640), "format": "RGB888"}))
    picam2.start()
    logging.info("[OK] Camera started.")

    # 2. Load the compiled YOLOv8 HEF model
    hef = HEF(MODEL_PATH)
    logging.info(f"[OK] Model {MODEL_PATH} successfully loaded.")

    # 3. Configure context and streams for the Hailo chip
    with VDevice() as target:
        net_group = target.configure(hef, ConfigureParams.create_from_hef(hef, interface=HailoStreamInterface.PCIe))[0]
        input_vstreams_params = InputVStreamParams.make(net_group, format_type=FormatType.UINT8)
        output_vstreams_params = OutputVStreamParams.make(net_group, format_type=FormatType.FLOAT32)

        with net_group.activate(net_group.create_params()), \
                InferVStreams(net_group, input_vstreams_params, output_vstreams_params) as infer_vstreams:
            logging.info("[SUCCESS] Monitoring started. AI analysis interval set to 3 seconds.")

            last_check = 0.0
            check_interval = 3.0  # Check interval in seconds

            try:
                while True:
                    # Step A: Capture a frame on every cycle to keep the buffer fresh
                    frame = picam2.capture_array()
                    current_time = time.time()

                    # Step B: Check if 10 seconds have passed since the last analysis
                    if current_time - last_check >= check_interval:

                        input_data = np.expand_dims(frame, axis=0).astype(np.uint8)
                        infer_results = infer_vstreams.infer(input_data)

                        nms_output = infer_results['yolov8s/yolov8_nms_postprocess']
                        coco_classes = nms_output[0]
                        person_detections = coco_classes[0]

                        best_confidence = float(np.max(person_detections[:, 4])) if person_detections.shape[0] > 0 else 0.0
                        # Step D: Action if human is detected
                        if best_confidence > 0.5:
                            logging.info(f"PERSON IN FRAME! Confidence: {best_confidence * 100:.1f}%")
                            verdict = classify_person_optimized(frame, "known_faces")
                            send_detection_to_server(frame, best_confidence, verdict)
                        else:
                            logging.info(f"Check: clear.")

                        last_check = current_time
            except KeyboardInterrupt:
                logging.info("[EXIT] Program stopped by user.")
            finally:
                picam2.stop()
                logging.info("[OK] Camera disconnected. Task complete.")


if __name__ == "__main__":
    main()