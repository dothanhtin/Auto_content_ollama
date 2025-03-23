from motor.motor_asyncio import AsyncIOMotorClient

# Kết nối MongoDB
MONGO_URI = "mongodb+srv://admin:PMP6nxCJ2RsYTyYm@fb-datamanagement-01.bml5n2k.mongodb.net/?retryWrites=true&w=majority"
DB_NAME = "FacebookTool"
COLLECTION_NAME = "SiteManageCollection"

mongoClient = AsyncIOMotorClient(MONGO_URI)
db = mongoClient[DB_NAME]
collection = db[COLLECTION_NAME]


def get_site_by_id(id: int):
    try:
        site = collection.find_one({"id": id})  # Tìm theo trường id trong document

        if site:
            site["_id"] = str(site["_id"])  # Chuyển ObjectId về string (nếu cần)
            return site
        else:
            return None  # Không tìm thấy
    except Exception as e:
        return {"error": str(e)}  # Xử lý lỗi