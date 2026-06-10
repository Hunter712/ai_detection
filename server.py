import os
import logging
import aiofiles
import asyncio
import httpx  # Import asynchronous HTTP client
from fastapi import FastAPI, File, UploadFile, BackgroundTasks

app = FastAPI()
logging.basicConfig(
    filename='server.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)

UPLOAD_DIR = "saved_photos"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# --- TELEGRAM SETTINGS ---
# Replace with your actual credentials
TELEGRAM_BOT_TOKEN = ""
TELEGRAM_CHAT_IDS = ["", ""]  # Your personal chat ID or group ID


async def send_to_single_chat(client: httpx.AsyncClient, url: str, chat_id: str, caption: str, filename: str,
                              file_content: bytes):
    data = {"chat_id": chat_id, "caption": caption, "parse_mode": "HTML"}
    files = {"photo": (filename, file_content, "image/jpeg")}
    try:
        response = await client.post(url, data=data, files=files, timeout=10)
        if response.status_code != 200:
            logging.error(f"[TG ERROR] Failed for {chat_id}: {response.status_code} - {response.text}")
        else:
            logging.info(f"[TG SUCCESS] Photo sent to chat {chat_id}")
    except Exception as e:
        logging.error(f"[TG ERROR] Connection failed for chat {chat_id}: {e}")


async def send_photo_to_telegram(file_path: str, caption: str):
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendPhoto"
    filename = os.path.basename(file_path)

    async with httpx.AsyncClient() as client:
        try:
            async with aiofiles.open(file_path, "rb") as f:
                file_content = await f.read()
            tasks = [
                send_to_single_chat(client, url, chat_id, caption, filename, file_content)
                for chat_id in TELEGRAM_CHAT_IDS
            ]

            await asyncio.gather(*tasks)

        except Exception as e:
            logging.error(f"[SYSTEM ERROR] Failed to read file or broadcast to Telegram: {e}")

@app.post("/items/")
async def create_item(background_tasks: BackgroundTasks,photo: UploadFile = File(...)):
    file_path = os.path.join(UPLOAD_DIR, photo.filename)

    # 1. Save the file locally on the server
    async with aiofiles.open(file_path, "wb") as f:
        content = await photo.read()
        await f.write(content)

    # Trigger the sending process (FastAPI waits for this before responding to client)
    background_tasks.add_task(send_photo_to_telegram, file_path, photo.filename)

    return {
        "message": "saved and sent to telegram",
        "filename": photo.filename
    }