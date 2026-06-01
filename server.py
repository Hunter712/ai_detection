import os
import aiofiles
import httpx  # Import asynchronous HTTP client
from fastapi import FastAPI, File, UploadFile

app = FastAPI()

UPLOAD_DIR = "saved_photos"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# --- TELEGRAM SETTINGS ---
# Replace with your actual credentials
TELEGRAM_BOT_TOKEN = ""
TELEGRAM_CHAT_IDS = ["", ""]  # Your personal chat ID or group ID


async def send_photo_to_telegram(file_path: str, caption: str):
    """Asynchronous function to send a photo to Telegram"""
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendPhoto"

    async with httpx.AsyncClient() as client:
        # Read the saved file asynchronously
        async with aiofiles.open(file_path, "rb") as f:
            file_content = await f.read()

        # Prepare multipart/form-data request
        files = {"photo": (os.path.basename(file_path), file_content)}
        for chat_id in TELEGRAM_CHAT_IDS:
            data = {"chat_id": chat_id, "caption": caption}
            try:
                response = await client.post(url, data=data, files=files, timeout=10)
                if response.status_code != 200:
                    print(f"[TG ERROR] Failed for {chat_id}: {response.status_code}")
                else:
                    print(f"[TG SUCCESS] Photo sent to chat {chat_id}")
            except Exception as e:
                print(f"[TG ERROR] Connection failed for chat {chat_id}: {e}")


@app.post("/items/")
async def create_item(photo: UploadFile = File(...)):
    file_path = os.path.join(UPLOAD_DIR, photo.filename)

    # 1. Save the file locally on the server
    async with aiofiles.open(file_path, "wb") as f:
        content = await photo.read()
        await f.write(content)

    # 2. Format the message text and send to Telegram
    # The filename (e.g., person_110333_01062026_52.9%.jpg) provides context
    tg_caption = f"🚨 **Person detected!**\n📄 File: `{photo.filename}`"

    # Trigger the sending process (FastAPI waits for this before responding to client)
    await send_photo_to_telegram(file_path, tg_caption)

    return {
        "message": "saved and sent to telegram",
        "filename": photo.filename
    }