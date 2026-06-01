import os
import aiofiles
from fastapi import FastAPI, File, UploadFile

app = FastAPI()

UPLOAD_DIR = "saved_photos"
os.makedirs(UPLOAD_DIR, exist_ok=True)


@app.post("/items/")
async def create_item(photo: UploadFile = File(...)):
    file_path = os.path.join(UPLOAD_DIR, photo.filename)

    async with aiofiles.open(file_path, "wb") as f:
        content = await photo.read()
        await f.write(content)

    return {"message": "saved", "filename": photo.filename}