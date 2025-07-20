from motor.motor_asyncio import AsyncIOMotorClient
from contextlib import asynccontextmanager

# databaseconnection.py
from motor.motor_asyncio import AsyncIOMotorClient
from contextlib import asynccontextmanager

MONGO_URI = "mongodb+srv://admin:PMP6nxCJ2RsYTyYm@fb-datamanagement-01.bml5n2k.mongodb.net/?retryWrites=true&w=majority"
DB_NAME = "AutoContent"
COLLECTION_NAME = "SiteManageCollection"

@asynccontextmanager
async def get_collection():
    client = AsyncIOMotorClient(MONGO_URI)
    try:
        db = client[DB_NAME]
        collection = db[COLLECTION_NAME]
        yield collection
    finally:
        client.close()

@asynccontextmanager
async def get_mongo_client():
    client = AsyncIOMotorClient(MONGO_URI)
    try:
        yield client
    finally:
        client.close()

async def get_site_by_id(id: int):
    async with get_mongo_client() as client:
        db = client[DB_NAME]
        collection = db[COLLECTION_NAME]
        try:
            site = await collection.find_one({"id": id})
            if site:
                site["_id"] = str(site["_id"])
                return site
            else:
                return None
        except Exception as e:
            return {"error": str(e)}