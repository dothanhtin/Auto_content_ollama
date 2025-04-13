from motor.motor_asyncio import AsyncIOMotorClient
import asyncio

# Kết nối MongoDB
MONGO_URI = "mongodb+srv://admin:PMP6nxCJ2RsYTyYm@fb-datamanagement-01.bml5n2k.mongodb.net/?retryWrites=true&w=majority"
DB_NAME = "AutoContent"
COLLECTION_NAME = "SiteManageCollection"

mongoClient = AsyncIOMotorClient(MONGO_URI)
db = mongoClient[DB_NAME]
collection = db[COLLECTION_NAME]


async def get_site_by_id(id: int):
    try:
        site = await collection.find_one({"id": id})  # Sử dụng await
        if site:
            site["_id"] = str(site["_id"])
            return site
        else:
            return None
    except Exception as e:
        return {"error": str(e)}