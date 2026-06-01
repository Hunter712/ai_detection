import time
import numpy as np
import cv2
import requests
from picamera2 import Picamera2
from hailo_platform import (HEF, VDevice, HailoStreamInterface, ConfigureParams,
                            InputVStreamParams, OutputVStreamParams, FormatType, InferVStreams)

# Configuration: Change this to your actual backend server IP and port
SERVER_URL = ""


def send_detection_to_server(frame, confidence):
    """
    Encodes the frame in-memory and sends ONLY the photo file to the server.
    The filename contains the timestamp and confidence score.
    """
    # 1. Format elements for the filename
    timestamp_file = time.strftime('%H%M%S_%d%m%Y')
    conf_percent = f"{confidence * 100:.1f}"
    filename = f"person_{timestamp_file}_{conf_percent}%.jpg"

    # 2. Encode the frame to JPEG directly in RAM
    success, encoded_image = cv2.imencode('.jpg', frame)
    if not success:
        print(f"[{time.strftime('%H:%M:%S')}] ❌ Failed to encode image in memory.")
        return

    try:
        # Prepare only the file, no text fields (payload) included
        files = {'photo': (filename, encoded_image.tobytes(), 'image/jpeg')}

        print(f"[{time.strftime('%H:%M:%S')}] 📡 Sending photo to server...")
        response = requests.post(SERVER_URL, files=files, timeout=5)

        if response.status_code == 200:
            print(f"[{time.strftime('%H:%M:%S')}] 🚀 Server sync successful: {response.text}")
        else:
            print(f"[{time.strftime('%H:%M:%S')}] ⚠️ Server returned status code: {response.status_code}")

    except requests.exceptions.RequestException as e:
        print(f"[{time.strftime('%H:%M:%S')}] ❌ Failed to connect to server: {e}")


def main():
    # 1. Initialize and start the camera
    picam2 = Picamera2()
    picam2.configure(picam2.create_video_configuration(main={"size": (640, 640), "format": "RGB888"}))
    picam2.start()
    print("[OK] Camera started.")

    # 2. Load the compiled YOLOv8 HEF model
    model_path = "/usr/local/hailo/resources/models/hailo8l/yolov8s.hef"
    hef = HEF(model_path)
    print(f"[OK] Model {model_path} successfully loaded.")

    # 3. Configure context and streams for the Hailo chip
    with VDevice() as target:
        net_group = target.configure(hef, ConfigureParams.create_from_hef(hef, interface=HailoStreamInterface.PCIe))[0]
        input_vstreams_params = InputVStreamParams.make(net_group, format_type=FormatType.UINT8)
        output_vstreams_params = OutputVStreamParams.make(net_group, format_type=FormatType.FLOAT32)

        with net_group.activate(net_group.create_params()), \
                InferVStreams(net_group, input_vstreams_params, output_vstreams_params) as infer_vstreams:

            print("[SUCCESS] Monitoring started. AI analysis interval set to 10 seconds.")
            print("--------------------------------------------------")

            last_check = 0.0
            check_interval = 10.0  # Check interval in seconds

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

                        human_found = False
                        best_confidence = 0.0

                        # Step C: Check if any reliable person detections exist
                        if person_detections.shape[0] > 0:
                            for detection in person_detections:
                                confidence = detection[4]

                                if confidence > 0.5:
                                    human_found = True
                                    if confidence > best_confidence:
                                        best_confidence = confidence

                        # Step D: Action if human is detected
                        if human_found:
                            print(
                                f"[{time.strftime('%H:%M:%S')}] 🚨 PERSON IN FRAME! Confidence: {best_confidence * 100:.1f}%")
                            send_detection_to_server(frame, best_confidence)
                        else:
                            print(f"[{time.strftime('%H:%M:%S')}] Check: clear.")

                        last_check = current_time

            except KeyboardInterrupt:
                print("\n[EXIT] Program stopped by user.")
            finally:
                picam2.stop()
                print("[OK] Camera disconnected. Task complete.")


if __name__ == "__main__":
    main()