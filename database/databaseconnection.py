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