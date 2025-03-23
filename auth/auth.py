from fastapi import FastAPI, Request, Depends, HTTPException
import redis
import httpx
import config

REDIS_HOST = config.REDIS_HOST
REDIS_PORT = config.REDIS_PORT
REDIS_PASSWORD = config.REDIS_PASSWORD
TOKEN_KEY = config.TOKEN_KEY
redis_client = redis.Redis(
    host=REDIS_HOST,
    port=REDIS_PORT,
    password=REDIS_PASSWORD,
    db=0,
    decode_responses=True
)

VERIFY_URL = config.VERIFY_URL
CACHE_EXPIRY = config.CACHE_EXPIRY


async def token_auth(request: Request):
    auth_header = request.headers.get("Authorization")
    if not auth_header or not auth_header.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Token missing or invalid")
    
    token = auth_header.split("Bearer ")[1]
    
    checked = await verify_token(token)  # ✅ Được phép dùng await trong async function
    
    if not checked:
        raise HTTPException(status_code=401, detail="Invalid token")
    
async def verify_token(token: str) -> bool:
    try:
        cache_key = f"token:{token}"
        cached_result = redis_client.get(cache_key)
        if cached_result:
            return cached_result == "1"

        async with httpx.AsyncClient() as client:
            response = await client.post(
                VERIFY_URL, 
                json={"token": token},
                headers={"Authorization": f"Bearer {token}"}
            )

        if response.status_code == 200:
            result = response.json()
            is_valid = result == 1  # API trả về {"valid": 1} nếu hợp lệ
            redis_client.setex(cache_key, CACHE_EXPIRY, "1" if is_valid else "0")
            return is_valid
        
        return False  # Trả về False nếu API verify lỗi
    except Exception as e:
        print(f"Error verifying token: {e}")
        return False