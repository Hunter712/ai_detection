import time
import logging
import requests
from gpiozero import MotionSensor

PIR_PIN = 17
CAMERA_PI_URL = ""
COOLDOWN_TIME = 3

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)


def send_trigger():
    try:
        response = requests.post(CAMERA_PI_URL, timeout=1.5)
        if response.status_code != 200:
            logging.warning(f"Camera server returned status: {response.status_code}")
    except requests.exceptions.RequestException as e:
        logging.error(f"Failed to connect to camera server: {e}")


def main():
    sensor = MotionSensor(PIR_PIN, queue_len=1, threshold=0.5)

    while True:
        sensor.wait_for_motion()
        send_trigger()
        #sensor.wait_for_no_motion()
        time.sleep(COOLDOWN_TIME)



if __name__ == "__main__":
    main()