# config.py
import os
from dotenv import load_dotenv

load_dotenv()
# --- API Configurations ---
LOCAL_API_URL = os.getenv("LOCAL_API_URL")
OPENROUTER_API_URL = os.getenv("OPENROUTER_API_URL")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
USE_OPENROUTER_API = os.getenv("USE_OPENROUTER_API") == "True"

# --- Redis Configuration ---
REDIS_HOST = os.getenv("REDIS_HOST")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))  # Default là 6379 nếu không có
REDIS_PASSWORD = os.getenv("REDIS_PASSWORD")
TOKEN_KEY = os.getenv("TOKEN_KEY")

# --- WordPress Configuration ---
WP_DOMAIN_URL = os.getenv("WP_DOMAIN_URL")
WP_URL = os.getenv("WP_URL")
WP_USERNAME = os.getenv("WP_USERNAME")
WP_PASSWORD = os.getenv("WP_PASSWORD")

# --- Cloudinary Configuration ---
CLOUDINARY_CLOUD_NAME = os.getenv("CLOUDINARY_CLOUD_NAME")
CLOUDINARY_API_KEY = os.getenv("CLOUDINARY_API_KEY")
CLOUDINARY_API_SECRET = os.getenv("CLOUDINARY_API_SECRET")

# --- YouTube & Unsplash Configuration ---
YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")
UNSPLASH_API_KEY = os.getenv("UNSPLASH_API_KEY")
IMAGE_API_KEY = os.getenv("IMAGE_API_KEY")
IMAGE_BASE_URL = os.getenv("IMAGE_BASE_URL")

# --- HuggingFace API ---
HF_API_KEY = os.getenv("HF_API_KEY")

LOGIN_URL = os.getenv("LOGIN_URL")
VERIFY_URL=os.getenv("VERIFY_URL")
CACHE_EXPIRY=os.getenv("CACHE_EXPIRY")
