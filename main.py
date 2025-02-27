import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from wordpress_xmlrpc import Client, WordPressPost
from wordpress_xmlrpc.methods.posts import NewPost
from wordpress_xmlrpc.methods.taxonomies import GetTerms
from wordpress_xmlrpc.methods.media import UploadFile
import xmlrpc.client
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from google_auth_oauthlib.flow import InstalledAppFlow
from huggingface_hub import InferenceClient
import cloudinary
import cloudinary.uploader
import io
import os
import datetime
import redis
import json


app = FastAPI()
class KeywordRequest(BaseModel):
    keyword: str

# Cấu hình API cục bộ của Ollama
LOCAL_API_URL = "http://localhost:11434/api/generate"  # Endpoint API cục bộ
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"
# OPENROUTER_API_KEY = "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"  # Deepseekfree_R1_apikey
# OPENROUTER_API_KEY = "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"  # Deepseekfree_V3_apikey
OPENROUTER_API_KEY = "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"  # Google flash thinking apikey

# Cấu hình Redis với mật khẩu
REDIS_HOST = "redis-19730.c82.us-east-1-2.ec2.redns.redis-cloud.com"  # Đổi thành địa chỉ Redis nếu cần
REDIS_PORT = 19730
REDIS_PASSWORD = "HHVwCTyETeYR7UVeidXNoavEWRiWjcYN"  # Nhập mật khẩu Redis của bạn
TOKEN_KEY = "wordpress_token"

# Kết nối Redis với mật khẩu
redis_client = redis.Redis(
    host=REDIS_HOST,
    port=REDIS_PORT,
    password=REDIS_PASSWORD,  # Thêm mật khẩu
    db=0,
    decode_responses=True
)


# Tùy chọn sử dụng API (True = OpenRouter, False = Ollama Local)
USE_OPENROUTER_API = True

# Cấu hình Wordpress để tự đăng bài
wp_domain_url = "https://niceplanet.xyz"
wp_url = "https://niceplanet.xyz/xmlrpc.php"
wp_username = "tindtadmin"
wp_password = "Ss123456789@"
wp_client = Client(wp_url, wp_username, wp_password)

# Cấu hình Cloudinary
cloudinary.config(
    cloud_name="kittykittenewordpress",
    api_key="484342665732471",
    api_secret="hC4dZwbrl5-k-V8biHf1oa3k974"
)

# Hàm tạo ảnh, upload lên Cloudinary, cập nhật Featured Image trong WordPress
def generate_and_upload_image(prompt, model, post_id, wordpress_url, wordpress_token):
    try:
        # Khởi tạo Hugging Face AI client
        client = InferenceClient(
            provider="fal-ai",
            api_key="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"
        )

        # Gọi API tạo ảnh từ văn bản
        image = client.text_to_image(prompt, model=model)

        # Lưu ảnh tạm vào file
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        temp_image_path = f"generated_image_{timestamp}.jpg"
        image.save(temp_image_path, format="JPEG")

        # Upload ảnh lên Cloudinary
        upload_result = cloudinary.uploader.upload(temp_image_path, folder="ai_generated_images")

        # Lấy URL của ảnh sau khi upload
        image_url = upload_result.get("secure_url")
        print(f"✅ Image uploaded: {image_url}")

        # Cập nhật Featured Image trong WordPress qua FIFU API
        wordpress_api_url = f"{wordpress_url}/wp-json/fifu/v2/image"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {wordpress_token}"
        }
        data = {"id": post_id, "src": image_url}

        response = requests.post(wordpress_api_url, json=data, headers=headers)

        if response.status_code == 200:
            print("✅ Featured Image updated successfully!")
        else:
            print("❌ Failed to update Featured Image:", response.text)

        # Xóa file tạm sau khi hoàn tất
        if os.path.exists(temp_image_path):
            os.remove(temp_image_path)
            print("🗑️ Temporary file deleted.")

        return image_url

    except Exception as e:
        print("❌ Error:", str(e))
        return None


# Hàm lấy token từ WordPress
def get_wordpress_token(username, password):
    url = "https://niceplanet.xyz/wp-json/jwt-auth/v1/token"
    headers = {"Content-Type": "application/json"}
    data = {"username": username, "password": password}

    response = requests.post(url, json=data, headers=headers)

    if response.status_code == 200 and "token" in response.json():
        token = response.json().get("token")
        print("✅ Token retrieved successfully!")

        # Lưu token vào Redis với thời gian hết hạn 1 giờ
        redis_client.setex(TOKEN_KEY, 3600, token)
        return token

    print("❌ Failed to get token:", response.text)
    return None

# Hàm kiểm tra token có hợp lệ không
def is_token_valid(token):
    url = "https://niceplanet.xyz/wp-json/jwt-auth/v1/token/validate"
    headers = {"Authorization": f"Bearer {token}"}

    response = requests.post(url, headers=headers)

    if response.status_code == 200:
        result = response.json()
        return result.get("success", False)
    
    return False

# Hàm lấy token hợp lệ từ Redis hoặc làm mới nếu hết hạn
def get_valid_token(username, password):
    token = redis_client.get(TOKEN_KEY)

    if token:
        print("🔍 Checking token validity...")
        if is_token_valid(token):
            print("✅ Token is valid.")
            return token
        else:
            print("❌ Token is invalid or expired. Refreshing...")

    # Lấy token mới nếu token cũ hết hạn
    return get_wordpress_token(username, password)


# Cấu hình Youtube API
youtube_api_key = "AIzaSyC9TRScaHRzBKbuRjyeyjSZceDxWLhnvX8"
your_groq_api_key = "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"

# Cấu hình Unsplash API để lấy ảnh tự động
image_api_key = "WCHwX4o62UMcUKL24mc-zbWr3EFgzkufKm5U3pf0OwY"
image_base_url = "https://api.unsplash.com/search/photos"


# Hàm gọi API Ollama cục bộ
def call_local_ollama(prompt, model="llama3:8b"):
    data = {
        "model": model,
        "prompt": prompt,
        "stream": False
    }
    try:
        response = requests.post(LOCAL_API_URL, json=data)
        if response.status_code == 200:
            response_data = response.json()  # Parse JSON
            return response_data.get("response", "No response found")  # Lấy nội dung từ trường "response"
        else:
            print(f"Error: {response.status_code}, {response.text}")
            return None
    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}")
        return None

# Hàm gọi AI từ API phù hợp
def call_ai_model(prompt, model="llama3:8b"):
    """
    Gọi AI từ API được chọn (Ollama hoặc OpenRouter AI).
    
    Parameters:
        prompt (str): Nội dung prompt gửi đến AI.
        model (str): Tên model sử dụng (chỉ áp dụng cho Ollama).

    Returns:
        str: Phản hồi từ AI.
    """
    if USE_OPENROUTER_API:
        headers = {
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json"
        }
        data = {
            "model": "google/gemini-2.0-flash-thinking-exp:free",  # Model OpenRouter
            "messages": [{"role": "user", "content": [{"type": "text", "text":prompt}]}]
        }
        api_url = OPENROUTER_API_URL
    else:
        data = {
            "model": model,  # Model Ollama (local)
            "prompt": prompt,
            "stream": False
        }
        api_url = LOCAL_API_URL

    try:
        response = requests.post(api_url, json=data, headers=headers if USE_OPENROUTER_API else None)
        if response.status_code == 200:
            response_data = response.json()
            return response_data.get("response", response_data.get("choices", [{}])[0].get("message", {}).get("content", "No response found"))
        else:
            print(f"Error: {response.status_code}, {response.text}")
            return None
    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}")
        return None

# 0. Xác định vai trò SEO analytics cho AI
def confirm_seo_analytics():
    prompt=f"""
        You are a content strategist specializing in SEO. Your task is to refine content strategy to attract organic traffic, improve search engine rankings, and deeply resonate with the target audience.
        Please follow these guidelines while analyzing and enhancing the content:
        ●	Relevance: Ensure all content is directly relevant to the website's core topics, products, or services.
        ●	Depth & Value: Prioritize in-depth, informative content that genuinely addresses user needs and questions.
        ●	Search Intent Match: Carefully align the content with what users are actually searching for (e.g., informational articles, product comparisons, how-to guides).
        ●	Target Audience Understanding: Craft content that speaks directly to the interests, pain points, and language of the ideal customer.
        ●	Trends & Freshness: Incorporate trending topics or emerging industry news when relevant, while keeping content up-to-date.
        ●	Semantic Richness: Use a variety of related terms and phrases naturally throughout the content to capture a wider range of search queries.
        ●	Readability & Engagement: Write in a clear, engaging style that keeps readers on the page.
        Please present your findings in a structured format, including the following:
        ●	Content Area: Identify specific pages or sections of content.
        ●	Current Status: Briefly assess the content's strengths and weaknesses.
        ●	Suggested Improvements: Provide actionable recommendations for improving relevance, depth, search intent match, audience appeal, etc.
        ●	SEO Considerations: Highlight opportunities to optimize title tags, meta descriptions, header tags, and internal linking for relevant keywords.
        ●	Additional Notes: Any further insights on content promotion, link building, or other strategies to boost visibility.
        Feel free to use any relevant content analysis and SEO tools to aid your assessment. Remember, the goal is to create a content strategy that not only ranks well but truly resonates with readers and drives meaningful engagement.
        Do you understand these revised instructions? You must response YES or NO. If so, acknowledge and await my next prompt.
    """
    return call_ai_model(prompt)

# 1. Phân tích chiến lược nội dung
def analyze_content(keyword):
    prompt = f"""
    You are a content strategist specializing in SEO. Create an outline for a blog targeting the keyword '{keyword}'. 
    Include an optimized title, meta description, H2 sections, and word count for each section.
    """
    return call_ai_model(prompt)

# 2. Tìm từ khóa phụ
def find_secondary_keywords(keyword):
    """
    Identifies secondary, NLP, and LSI keywords related to the given keyword.

    Parameters:
        keyword (str): The main keyword to analyze.

    Returns:
        dict: A dictionary containing:
            - top_3_secondary_keywords: List of the top 3 secondary keywords.
            - top_3_nlp_keywords: List of the top 3 NLP keywords.
            - top_3_lsi_keywords: List of the top 3 LSI keywords.
            - concatenated_secondary_keywords: A string of top 3 secondary keywords separated by commas.
            - concatenated_nlp_lsi_keywords: A string of top 3 NLP and LSI keywords separated by commas.
    """
    prompt = f"""
    Identify secondary, NLP, and LSI keywords related to '{keyword}' that are relevant and have low competition but decent search volume.
    with main format is easy retrive by coding such as:
        **Secondary Keywords**
        1.
        2.
        ...
        **NLP Keywords**
        1.
        2.
        ...
        **LSI Keywords**
        1.
        2.
        ....
    """
    # Gọi hàm xử lý của mô hình AI
    result_text = call_ai_model(prompt)

    try:
        import re

        # Định nghĩa các danh sách kết quả
        sections = {
            "Secondary Keywords": [],
            "NLP Keywords": [],
            "LSI Keywords": []
        }

        # Tìm kiếm các phần từ khóa bằng regex
        for section_name in sections.keys():
            pattern = rf"\*\*{section_name}\*\*.*?(?=(\*\*|$))"  # Tìm phần tiêu đề và nội dung
            match = re.search(pattern, result_text, re.DOTALL)
            if match:
                content = match.group()  # Lấy nội dung phần này
                keywords = re.findall(r"\d+\.\s+(.*)", content)  # Tìm các từ khóa dạng "1. từ khóa"
                sections[section_name] = keywords

        # Lấy top 3 từ mỗi danh sách
        top_3_secondary_keywords = sections["Secondary Keywords"][:3]
        top_3_nlp_keywords = sections["NLP Keywords"][:3]
        top_3_lsi_keywords = sections["LSI Keywords"][:3]

        # Nối các từ khóa thành chuỗi
        concatenated_secondary_keywords = ", ".join(top_3_secondary_keywords)
        concatenated_nlp_lsi_keywords = ", ".join(top_3_nlp_keywords + top_3_lsi_keywords)

        return {
            "top_3_secondary_keywords": top_3_secondary_keywords,
            "top_3_nlp_keywords": top_3_nlp_keywords,
            "top_3_lsi_keywords": top_3_lsi_keywords,
            "concatenated_secondary_keywords": concatenated_secondary_keywords,
            "concatenated_nlp_lsi_keywords": concatenated_nlp_lsi_keywords
        }

    except Exception as e:
        # Xử lý lỗi nếu không thể phân tích kết quả trả về
        print(f"Error processing secondary keywords: {e}")
        return {
            "top_3_secondary_keywords": [],
            "top_3_nlp_keywords": [],
            "top_3_lsi_keywords": [],
            "concatenated_secondary_keywords": "",
            "concatenated_nlp_lsi_keywords": ""
        }




# 3. Tìm từ khóa dài
def find_longterm_keywords(keyword):
    """
    Finds long-tail keywords related to the given keyword.

    Parameters:
        keyword (str): The main keyword to analyze.

    Returns:
        dict: A dictionary containing the top 3 long-tail keywords and a concatenated string of all keywords.
    """
    prompt = f"""
    Identify long-tail keywords related to '{keyword}' that are relevant and have low competition but decent search volume.
    Arrange them in separate lists.
    """
    # Gọi hàm xử lý của mô hình để lấy kết quả từ prompt
    result = call_ai_model(prompt)

    # Giả sử `result` trả về một cấu trúc JSON dạng:
    # {
    #   "long_tail_keywords": ["keyword1", "keyword2", "keyword3", ...]
    # }
    all_keywords = []

    try:
        # Parse danh sách từ khóa dài từ kết quả
        long_tail_keywords = result.get("long_tail_keywords", [])
        
        # Gộp tất cả từ khóa dài thành một danh sách
        all_keywords = long_tail_keywords

        # Chọn 3 từ khóa dài tốt nhất (giả sử các từ khóa đầu danh sách là tốt nhất)
        top_3_keywords = all_keywords[:3]

        # Nối tất cả từ khóa dài thành một chuỗi, ngăn cách bởi dấu phẩy
        concatenated_keywords = ", ".join(all_keywords)

        # Trả về kết quả dưới dạng dictionary
        return {
            "top_3_longtail_keywords": top_3_keywords,
            "concatenated_keywords": concatenated_keywords
        }

    except Exception as e:
        # Xử lý lỗi nếu kết quả không đúng định dạng mong đợi
        print(f"Error processing long-tail keywords: {e}")
        return {
            "top_3_longtail_keywords": [],
            "concatenated_keywords": ""
        }


# 4. Phân tích từ khóa từ các site
def analyze_site_keywords(keyword, site_list):
    """
    Analyzes the top-ranking pages of the given sites and identifies the targeted keywords.

    Parameters:
        keyword (str): The main keyword to analyze.
        site_list (list): A list of sites to analyze.

    Returns:
        dict: A dictionary containing the top 3 keywords for each site and a concatenated string of all keywords across all sites.
    """
    results = {}
    all_keywords = []

    for site in site_list:
        prompt = f"""
        Analyze the top-ranking pages of {site} and identify the keywords they are targeting for organic search traffic. 
        Focus on keywords that are relevant to '{keyword}' and have high search volume and moderate to low competition.
        """
        # Gọi hàm xử lý của mô hình để lấy kết quả từ prompt
        analysis_result = call_ai_model(prompt)

        # Giả sử `analysis_result` trả về một cấu trúc JSON dạng:
        # {
        #   "keywords": ["keyword1", "keyword2", "keyword3", ...]
        # }
        try:
            # Lấy danh sách từ khóa từ kết quả
            site_keywords = analysis_result.get("keywords", [])

            # Gộp tất cả từ khóa từ site vào danh sách chung
            all_keywords.extend(site_keywords)

            # Chọn 3 từ khóa tốt nhất từ site
            top_3_keywords = site_keywords[:3]

            # Lưu kết quả cho từng site
            results[site] = {
                "top_3_keywords": top_3_keywords,
                "all_site_keywords": site_keywords
            }

        except Exception as e:
            # Xử lý lỗi nếu kết quả không đúng định dạng mong đợi
            print(f"Error processing keywords for site {site}: {e}")
            results[site] = {
                "top_3_keywords": [],
                "all_site_keywords": []
            }

    # Nối tất cả từ khóa thành một chuỗi ngăn cách bởi dấu phẩy
    concatenated_keywords = ", ".join(all_keywords)

    return {
        "site_results": results,
        "concatenated_keywords": concatenated_keywords
    }


#5. Keyword trending
def find_trending_keywords_and_topics(keyword):
    """
    Finds trending keywords and topics related to the given keyword.

    Parameters:
        keyword (str): The main keyword to analyze.

    Returns:
        dict: A dictionary containing the top 3 trending keywords and a concatenated string of all keywords.
    """
    prompt = f"""
    Identify trending keywords and topics in '{keyword}' that have experienced a significant increase in search volume or popularity in recent weeks or months. 
    Consider factors like news events, social media trends, seasonal changes, and emerging technologies.
    """
    # Gọi hàm xử lý của mô hình để lấy kết quả từ prompt
    result = call_ai_model(prompt)

    # Giả sử `result` trả về một cấu trúc JSON dạng:
    # {
    #   "trending_keywords": ["keyword1", "keyword2", "keyword3", ...],
    #   "trending_topics": ["topic1", "topic2", "topic3", ...]
    # }
    all_keywords = []

    try:
        # Parse các danh sách từ khóa và chủ đề từ kết quả
        trending_keywords = result.get("trending_keywords", [])
        trending_topics = result.get("trending_topics", [])
        
        # Gộp tất cả từ khóa và chủ đề thành một danh sách duy nhất
        all_keywords = trending_keywords + trending_topics

        # Chọn 3 từ khóa tốt nhất (giả sử các từ khóa đầu danh sách là tốt nhất)
        top_3_keywords = all_keywords[:3]

        # Nối tất cả từ khóa thành một chuỗi, ngăn cách bởi dấu phẩy
        concatenated_keywords = ", ".join(all_keywords)

        # Trả về kết quả dưới dạng dictionary
        return {
            "top_3_trending_keywords": top_3_keywords,
            "concatenated_keywords": concatenated_keywords
        }

    except Exception as e:
        # Xử lý lỗi nếu kết quả không đúng định dạng mong đợi
        print(f"Error processing trending keywords: {e}")
        return {
            "top_3_trending_keywords": [],
            "concatenated_keywords": ""
        }


#6. Keyword with location
def find_local_keywords_and_phrases(keyword, location="US"):
    """
    Identifies local keywords and phrases related to the given keyword and location.

    Parameters:
        keyword (str): The main keyword to analyze.
        location (str): The target location (default is "US").

    Returns:
        dict: A dictionary containing the top 3 local keywords and a concatenated string of all local keywords.
    """
    prompt = f"""
    Identify local keywords and phrases that people in '{location}' are using to search for information like '{keyword}'. 
    Include keywords that mention the location, specific neighborhoods, landmarks, or nearby areas. 
    Also, consider keywords that reflect local search intent, such as "near me".
    """
    # Gọi hàm xử lý của mô hình AI để lấy kết quả từ prompt
    local_keywords_result = call_ai_model(prompt)

    # Giả sử `local_keywords_result` trả về một cấu trúc JSON dạng:
    # {
    #   "local_keywords": ["keyword1", "keyword2", "keyword3", ...]
    # }
    try:
        # Lấy danh sách từ khóa từ kết quả
        local_keywords = local_keywords_result.get("local_keywords", [])

        # Chọn 3 từ khóa tốt nhất
        top_3_keywords = local_keywords[:3]

        # Nối tất cả từ khóa thành một chuỗi ngăn cách bởi dấu phẩy
        concatenated_keywords = ", ".join(local_keywords)

        return {
            "top_3_keywords": top_3_keywords,
            "concatenated_keywords": concatenated_keywords
        }

    except Exception as e:
        # Xử lý lỗi nếu kết quả không đúng định dạng mong đợi
        print(f"Error processing local keywords: {e}")
        return {
            "top_3_keywords": [],
            "concatenated_keywords": ""
        }


#7. Create outline
def create_seo_content_outline(keyword):
    """
    Generates an SEO content outline using a given keyword.

    Parameters:
        keyword (str): The keyword or topic for which to generate an SEO content outline.

    Returns:
        str: The SEO content outline generated by the AI.
    """
    prompt = f"""
    Use Google search to look up the top 5 results and create an SEO content outline for '{keyword}'. 
    For each section, include the amount of words. 
    Also, include an optimized title, meta description, and H2s.
    """
    # Gọi hàm xử lý của mô hình (giả sử bạn có hàm `call_ai_model` để giao tiếp với mô hình AI)
    seo_outline = call_ai_model(prompt)
    return seo_outline


# 8. Tối ưu outline
def optimize_outline(outline, keyword):
    """
    Optimize the outline for a given keyword, adding a title and meta description.

    Parameters:
        outline (str): The original outline to optimize.
        keyword (str): The primary keyword to target.

    Returns:
        dict: A dictionary containing:
            - title: The optimized title.
            - meta_description: The optimized meta description.
            - optimized_outline: The original outline as returned by Ollama.
    """
    prompt = f"""
    Please optimize this outline to target the primary keyword '{keyword}' at a 3% density. 
    Use other keywords naturally. Add an FAQ section to Optimized Outline and provide an appropriate title and meta description.
    Format output:
        **Title:**...
        **Meta Description:**....
        **Optimized Outline:**...
    Outline: {outline}
    """
    result_text = call_ai_model(prompt)

    try:
        # Extract Title
        title_start = result_text.find("**Title:**")
        title_end = result_text.find("\n", title_start)
        title = result_text[title_start + 10:title_end].strip() if title_start != -1 else ""

        # Extract Meta Description
        meta_start = result_text.find("**Meta Description:**")
        meta_end = result_text.find("\n", meta_start)
        meta_description = (
            result_text[meta_start + 20:meta_end].strip() if meta_start != -1 else ""
        )

        # Extract Optimized Outline
        outline_start = result_text.find("**Optimized Outline:**")
        optimized_outline = (
            result_text[outline_start:].strip() if outline_start != -1 else result_text
        )

        return {
            "title": title,
            "meta_description": meta_description,
            "optimized_outline": optimized_outline,
        }

    except Exception as e:
        # Handle errors during processing
        print(f"Error processing outline: {e}")
        return {
            "title": "",
            "meta_description": "",
            "optimized_outline": outline,
        }



# 9. Viết nội dung
def write_content(outline, keyword, secondaryKeywords,LSIandNLPKeywords):
    prompt = f"""
    Start writing the content with {outline}, one section at a time, auto next section and combine all section to an completed article. Utilize the {keyword}, secondary keywords: {secondaryKeywords}, LSI and NLP keywords : {LSIandNLPKeywords} listed through out the content naturally. Also, maintain the word count specified for each section. Note that I want the content to be written like it was written by a subject matter expert, without fluff or jargon. The content you produce should also have low AI content detection scores and should reflect the same when passed through AI content detectors. Write in a [tone/style: informative] voice that engages the reader. The content should sound like it was written by a person, not a machine. Avoid clichés, jargon, and overly complex sentence structures.. Focus on originality. Do not plagiarize existing work. Ensure the facts and ideas presented are accurate and well-researched.
    output is only completed article to retrieve data easily to post wordpress.
    """
    return call_ai_model(prompt)

# Fetch relevant image
def get_image_url(query):
    headers = {"Authorization": f"Client-ID {image_api_key}"}
    params = {"query": query, "per_page": 1}
    response = requests.get(image_base_url, headers=headers, params=params)
    data = response.json()

    if data['results']:
        return data['results'][0]['urls']['regular']
    return None

def get_youtube_video_id(query, youtube_api_key):
    youtube = build("youtube", "v3", developerKey=youtube_api_key)

    try:
        response = youtube.search().list(
            q=query,
            part="id,snippet",
            maxResults=1,
            type="video",
        ).execute()

        if response["items"]:
            return response["items"][0]["id"]["videoId"]
            print("youtube is found!")
        return None
        print("youttube is not found!")

    except HttpError as error:
        print(f"An error occurred: {error}")
        return None
    
#format title    
def format_title(title):
    """
    Format a WordPress title by:
    - Removing '**' characters.
    - Removing any "(Word count: ...)" substring.

    Parameters:
        title (str): The original title.

    Returns:
        str: The formatted title.
    """
    import re
    # Remove '**' characters
    title = title.replace("**", "")
    # Remove "(Word count: ...)" using regex
    title = re.sub(r"\(Word count:.*?\)", "", title).strip()
    return title

#format content
import re

def format_content(content):
    """
    Format the content by:
    - Removing unnecessary phrases and lines.
    - Formatting titles marked with '**' as <h2>.
    - Removing "Section x" and any lines containing "Word count:".

    Parameters:
        content (str): The original content to format.

    Returns:
        str: The formatted content.
    """
    # List of phrases to completely remove (entire line will be removed)
    phrases_to_remove = [
        "Here is the completed article based on the optimized outline:",
        "**Optimized Outline: Volvo's Electric Hybrid Efforts**",
        "I hope this meets your requirements! Let me know if you need any further modifications."
    ]

    # Remove lines containing any phrase in `phrases_to_remove` or "Word count:"
    content_lines = []
    if content is not None:
        content_lines = content.splitlines()  
    filtered_lines = [
        line for line in content_lines
        if not any(phrase in line for phrase in phrases_to_remove) and "Word count:" not in line
    ]

    # Process the filtered lines
    processed_lines = []
    for line in filtered_lines:
        stripped_line = line.strip()

        # Remove "Section x:" (e.g., "Section 1:") from the beginning of the line
        stripped_line = re.sub(r"^Section\s+\d+:\s*", "", stripped_line)

        # Convert lines starting and ending with '**' to H2
        if stripped_line.startswith("**") and stripped_line.endswith("**"):
            h2_content = stripped_line.strip("**").strip()
            processed_lines.append(f"<h2>{h2_content}</h2>")
        else:
            processed_lines.append(stripped_line)

    # Join the processed lines into the final content
    formatted_content = "\n".join(processed_lines).strip()
    return formatted_content

@app.get("/status")
def status():
    return {"status": "API is running"}

@app.get("/test")
def test_api():
    return {"message": "This is a test response!"}


# API thực hiện toàn bộ quy trình SEO
@app.post("/write_seo_content")
def seo_pipeline(request: KeywordRequest):
    keyword = request.keyword.strip()
    if not keyword:
        keyword = "volvo cars electric hybrid"
    print(f"Từ khóa chính: {keyword}")

    confirm = confirm_seo_analytics()
    if "YES" not in confirm:
        raise HTTPException(status_code=400, detail="AI did not confirm SEO analytics role")
    
    # Bước 1: Phân tích và tạo outline
    outline = create_seo_content_outline(keyword)

    # Bước 2: Tìm từ khóa phụ, nsi, npl keywords
    secondary_keywords = find_secondary_keywords(keyword)

    # Bước 3: Tối ưu outline
    optimized_outline = optimize_outline(outline, keyword)
    print("\nOptimized Outline:\n", optimized_outline["optimized_outline"])

    # Bước 4: Viết nội dung
    content = write_content(optimized_outline["optimized_outline"], keyword,secondary_keywords["concatenated_secondary_keywords"],secondary_keywords["concatenated_nlp_lsi_keywords"])
    print("\nContent:\n", content)

    # Đăng bài lên wordpress
    post = WordPressPost()
    formatted_title  = format_title(optimized_outline["title"])
    post.title = formatted_title
    formatted_content = format_content(content)

    # Thêm video vào đầu bài viết
    video_id = get_youtube_video_id(keyword, youtube_api_key)
    if video_id:
        youtube_embed = f'<iframe width="560" height="315" src="https://www.youtube.com/embed/{video_id}" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>'
        blog_content = f"{youtube_embed}\n\n{formatted_content}"
        post.content = blog_content
    else:
        post.content = formatted_content
    '''post.terms_names = {
        'post_tag': tags,
        'category': categories,
    }'''

    #post.post_status = 'publish'
    post.post_status = 'draft'
    post_id = wp_client.call(NewPost(post))

    # Đăng ảnh 

    wp_token = get_valid_token(wp_username,wp_password);

    image_url = generate_and_upload_image(
        prompt=formatted_title,
        model="stabilityai/stable-diffusion-3.5-large",
        post_id=post_id,  # ID bài viết WordPress
        wordpress_url=wp_domain_url,
        wordpress_token=wp_token
    )

    print("🔗 Final Image URL:", image_url)
    
    return {
        "keyword": keyword,
        "outline": outline,
        "secondary_keywords": secondary_keywords,
        "optimized_outline": optimized_outline,
        "content": content
    }
