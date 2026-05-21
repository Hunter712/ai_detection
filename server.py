from fastapi import FastAPI
from pydantic import BaseModel
import aiofiles
from datetime import datetime

app = FastAPI()


class UserItem(BaseModel):
    event: str
    confidence: float
    timestamp: str


@app.post("/items/")
async def create_item(item: UserItem):
    dt_object = datetime.fromtimestamp(int(item.timestamp))
    readable_date = dt_object.strftime("%d.%m.%Y %H:%M:%S")
    event_name = item.event
    confidence_val = round(item.confidence, 2)

    log_line = f"{readable_date} | {event_name} | {confidence_val}\n"

    async with aiofiles.open("items_db.txt", mode="a", encoding="utf-8") as f:
        await f.write(log_line)

    return {"message": "saved", "data": item}