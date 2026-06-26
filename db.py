import os
from datetime import datetime, timezone
from motor.motor_asyncio import AsyncIOMotorClient
from bson import ObjectId

MONGO_URL = os.environ.get("MONGODB_URL", "mongodb://localhost:27017/anomaly_detection")
DB_NAME = "anomaly_detection"

client: AsyncIOMotorClient | None = None
available = False

if "serverSelectionTimeoutMS" not in MONGO_URL:
    sep = "&" if "?" in MONGO_URL else "?"
    MONGO_URL = f"{MONGO_URL}{sep}serverSelectionTimeoutMS=3000&connectTimeoutMS=3000&socketTimeoutMS=3000"

async def connect_db():
    global client, available
    try:
        client = AsyncIOMotorClient(MONGO_URL)
        await client.admin.command("ping")
        available = True
        print(f"MongoDB connected")
        try:
            await client[DB_NAME]["detections"].create_index("created_at", -1)
        except:
            pass  # index optional
    except Exception as e:
        msg = str(e).split("\n")[0][:80]
        print(f"MongoDB not available ({msg}). Running without persistence.")
        if client:
            client.close()
        client = None
        available = False

async def close_db():
    if client:
        client.close()

async def save_detection(record: dict) -> str | None:
    if not available or client is None:
        return None
    try:
        record["created_at"] = datetime.now(timezone.utc)
        result = await client[DB_NAME]["detections"].insert_one(record)
        rid = str(result.inserted_id)
        print(f"DB: saved detection {rid}")
        return rid
    except Exception as e:
        print(f"DB: save failed - {e}")
        return None

async def get_history(skip: int = 0, limit: int = 50) -> list:
    if not available or client is None:
        return []
    try:
        col = client[DB_NAME]["detections"]
        cursor = col.find().sort("created_at", -1).skip(skip).limit(limit)
        results = []
        async for doc in cursor:
            doc["id"] = str(doc.pop("_id"))
            if "created_at" in doc:
                doc["created_at"] = doc["created_at"].isoformat()
            results.append(doc)
        return results
    except:
        return []

async def count_detections() -> int:
    if not available or client is None:
        return 0
    try:
        return await client[DB_NAME]["detections"].count_documents({})
    except:
        return 0

async def delete_history(ids: list[str] | None = None):
    if not available or client is None:
        return
    try:
        col = client[DB_NAME]["detections"]
        if ids:
            obj_ids = [ObjectId(i) for i in ids]
            await col.delete_many({"_id": {"$in": obj_ids}})
        else:
            await col.delete_many({})
    except:
        pass

async def delete_one_detection(id: str) -> bool:
    if not available or client is None:
        return False
    try:
        result = await client[DB_NAME]["detections"].delete_one({"_id": ObjectId(id)})
        return result.deleted_count > 0
    except:
        return False

async def update_feedback(id: str, feedback: str) -> bool:
    if not available or client is None:
        return False
    try:
        result = await client[DB_NAME]["detections"].update_one(
            {"_id": ObjectId(id)},
            {"$set": {"feedback": feedback}}
        )
        return result.modified_count > 0
    except:
        return False
