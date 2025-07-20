import requests
from fastapi import FastAPI, Request, Depends, HTTPException, UploadFile as FastAPIUploadFile, File
from pydantic import BaseModel
from wordpress_xmlrpc import Client, WordPressPost
from wordpress_xmlrpc.methods.posts import NewPost
from wordpress_xmlrpc.methods.taxonomies import GetTerms
from wordpress_xmlrpc.methods.media import UploadFile
import xmlrpc.client
import io
import json
import config
import httpx
from fastapi.routing import APIRoute
import auth.auth as auth
from pyexcel_xlsx import get_data
from io import BytesIO
from typing import List
import database.databaseconnection as db
import functions.helpers  as helpers
from thirdparty.redisconnection import redis_client
from fastapi.middleware.cors import CORSMiddleware
import asyncio
import logging

app = FastAPI()
logger = logging.getLogger("uvicorn")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Hoặc cụ thể: ["http://127.0.0.1:5000"]
    allow_credentials=True,
    allow_methods=["*"],  # Cho phép tất cả các phương thức (GET, POST, ...)
    allow_headers=["*"],  # Cho phép tất cả các header
)

class KeywordRequest(BaseModel):
    keyword: str
    siteId: str
       
class ListKeywordRequest(BaseModel):
    keywords: List[str]
    siteId : str

# -------------------------
# Cấu hình các API & Service
# Cấu hình API
LOGIN_URL = config.LOGIN_URL
VERIFY_URL = config.VERIFY_URL
CACHE_EXPIRY = int(config.CACHE_EXPIRY)  # Cache token trong 1 giờ

# -------------------------
# API Ollama & OpenRouter
LOCAL_API_URL = config.LOCAL_API_URL
OPENROUTER_API_URL = config.OPENROUTER_API_URL
OPENROUTER_API_KEY = config.OPENROUTER_API_KEY
USE_THIRD_PARTY_API = config.USE_THIRD_PARTY_API



# Cấu hình WordPress
wp_url = config.WP_URL

# -------------------------
# Tích hợp toàn bộ quy trình vào một class pipeline
# -------------------------
class SEOContentPipeline:
    def __init__(self, keyword,siteId):
        self.keyword = keyword.strip() if keyword.strip() else "volvo cars electric hybrid"
        self.siteId = siteId
        self.context = {}
    
    def run_pipeline(self):
        # Bước 0: Xác nhận vai trò SEO Analytics
        confirm = helpers.confirm_seo_analytics()
        if "YES" not in confirm:
            raise Exception("AI did not confirm SEO analytics role")
        self.context["confirm"] = confirm

        # Bước 1: Tạo outline từ nội dung SEO
        outline = helpers.create_seo_content_outline(self.keyword)
        self.context["outline"] = outline

        # Bước 2: Tìm từ khóa phụ (Secondary, NLP, LSI)
        secondary_keywords = helpers.find_secondary_keywords(self.keyword)
        self.context["secondary_keywords"] = secondary_keywords

        # Bước 3: Tối ưu outline dựa trên từ khóa chính
        optimized_outline = helpers.optimize_outline(outline, self.keyword)
        self.context["optimized_outline"] = optimized_outline

        # Bước 4: Viết nội dung dựa trên outline tối ưu và từ khóa phụ
        content = helpers.write_content(
            optimized_outline.get("optimized_outline", ""),
            self.keyword,
            secondary_keywords.get("concatenated_secondary_keywords", ""),
            secondary_keywords.get("concatenated_nlp_lsi_keywords", "")
        )
        self.context["content"] = content

        # Bước 5: Đăng bài lên WordPress
        logger.info(f"Đăng bài lên WordPress!")
        post = WordPressPost()
        formatted_title = helpers.format_title(optimized_outline.get("title", self.keyword.title()))
        post.title = formatted_title
        formatted_content = helpers.format_content(content)
        # Thêm video YouTube nếu có
        video_id = helpers.get_youtube_video_id(self.keyword, config.YOUTUBE_API_KEY)
        if video_id:
            youtube_embed = f'<iframe width="560" height="315" src="https://www.youtube.com/embed/{video_id}" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>'
            blog_content = f"{youtube_embed}\n\n{formatted_content}"
            post.content = blog_content
        else:
            post.content = formatted_content
        post.post_status = 'draft'

        siteResult = get_site_by_id_sync(self.siteId)
        if not siteResult:
            logger.info("Site not found")
            raise Exception("Site not found")
        
        #wp_domain_url = siteResult["url"]
        wp_domain_url = siteResult.get("url")
        url = wp_domain_url + wp_url
        # wp_username = siteResult["username"]
        # wp_password = siteResult["password"]

        wp_username = siteResult.get("username")
        wp_password = siteResult.get("password")

        #Ghi log chỗ này
        logger.info(f"wp_domain_url: {wp_domain_url}, wp_url: {wp_url}, full_url: {url}")
        logger.info(f"wp_username: {wp_username}, wp_password: {wp_password}")

        wp_client = Client(url, wp_username, wp_password)
        post_id = wp_client.call(NewPost(post))
        self.context["post_id"] = post_id

        # Bước 6: Tạo ảnh, upload lên Cloudinary và cập nhật Featured Image trong WordPress
        wp_token = helpers.get_valid_token(wp_domain_url,wp_username, wp_password)
        image_url = helpers.generate_and_upload_image(
            prompt=formatted_title,
            model=config.IMAGE_MODEL_AI,
            post_id=post_id,
            wp_domain_url=wp_domain_url,
            wordpress_token=wp_token
        )
        self.context["image_url"] = image_url

        return {
            "keyword": self.keyword,
            "outline": outline,
            "secondary_keywords": secondary_keywords,
            "optimized_outline": optimized_outline,
            "content": content,
            "wordpress_post_id": post_id,
            "featured_image_url": image_url
        }

# -------------------------
# Các API endpoint
# -------------------------
async def get_token(login_payload: dict):
    """Lấy token từ Redis hoặc đăng nhập lại nếu hết hạn."""
    token = redis_client.get("auth_token")
    if token:
        return token

    # Gọi API để lấy token mới
    try:
        response = requests.post(LOGIN_URL, json=login_payload, timeout=5)
        if response.status_code == 200:
            token = response.json().get("token")
            if token:
                redis_client.setex("auth_token", CACHE_EXPIRY, token)  # Lưu token vào Redis
                return token
    except requests.RequestException:
        pass
    return None

def get_site_by_id_sync(id):
    # Lấy event loop hiện tại hoặc tạo mới
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    
    # Chạy hàm async và lấy kết quả
    siteId = int(id)
    return loop.run_until_complete(db.get_site_by_id(siteId))

@app.post("/get-token")
async def generate_token(login_payload: dict):
    """API để lấy token với thông tin đăng nhập động."""
    if not login_payload or "username" not in login_payload or "password" not in login_payload:
        raise HTTPException(status_code=400, detail="Missing username or password")

    #token = await get_token(login_payload)
    token = await asyncio.wait_for(get_token(login_payload), timeout=60)
    if token:
        return {"token": token}
    raise HTTPException(status_code=401, detail="Failed to authenticate")


@app.get("/status",dependencies=[Depends(auth.token_auth)])
def status():
    return {"status": "API is running"}

@app.get("/test",dependencies=[Depends(auth.token_auth)])
def test_api():
    return {"message": "This is a test response!"}

@app.post("/write_seo_content",dependencies=[Depends(auth.token_auth)])
def seo_pipeline(request: KeywordRequest):
    pipeline = SEOContentPipeline(request.keyword,request.siteId)
    try:
        result = pipeline.run_pipeline()
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/write_seo_content_bulk",dependencies=[Depends(auth.token_auth)])
def seo_pipeline_bulk(request: ListKeywordRequest):
    results = []
    for keyword in request.keywords:
        pipeline = SEOContentPipeline(keyword,request.siteId)
        try:
            result = pipeline.run_pipeline()
            results.append({keyword: result})
        except Exception as e:
            results.append({keyword: f"Error: {str(e)}"})
    
    return results

@app.post("/import_and_write_seo_content", dependencies=[Depends(auth.token_auth)], response_model=None)
def import_and_write_seo_content(
    excel_file: FastAPIUploadFile = File(...),
    siteId: str = "1"
):
    try:
        # Đọc file Excel bằng pyexcel_xlsx
        data = get_data(BytesIO(excel_file.file.read()))
        sheet = data[list(data.keys())[0]]  # Lấy sheet đầu tiên
        
        # Kiểm tra cột 'Keyword'
        if "Keyword" not in sheet[0]:  # sheet[0] là header
            raise HTTPException(status_code=400, detail="Excel file must contain a 'Keyword' column")
        
        keyword_col_idx = sheet[0].index("Keyword")
        keywords = [row[keyword_col_idx] for row in sheet[1:] if row[keyword_col_idx]]  # Lấy cột Keyword, bỏ header
        
        results = []
        for keyword in keywords:
            pipeline = SEOContentPipeline(keyword, siteId)
            try:
                result = pipeline.run_pipeline()
                results.append({keyword: result})
            except Exception as e:
                results.append({keyword: f"Error: {str(e)}"})
        
        return results
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/sites",dependencies=[Depends(auth.token_auth)])
async def get_sites():
    try:
        sites = []
        async for site in db.collection.find({}, {"_id": 0}):  # Không lấy _id
            sites.append(site)
        return {"sites": sites}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))