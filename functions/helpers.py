import requests
from thirdparty.redisconnection import redis_client
import config
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from google_auth_oauthlib.flow import InstalledAppFlow
from huggingface_hub import InferenceClient
import re
import datetime
import os
import cloudinary
import cloudinary.uploader

# -------------------------
# Các hàm hỗ trợ (helper functions)
# -------------------------

# Cấu hình Cloudinary
cloudinary.config(
    cloud_name=config.CLOUDINARY_CLOUD_NAME,
    api_key=config.CLOUDINARY_API_KEY,
    api_secret=config.CLOUDINARY_API_SECRET
)

def generate_with_cloudflare(prompt):
    url = "https://api.cloudflare.com/client/v4/accounts/eb9da718bb875b75af5a984d0cd76fe7/ai/run/@cf/stabilityai/stable-diffusion-xl-base-1.0"
    headers = {
        "Authorization": f"Bearer {config.CLOUDFLARE_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "prompt": prompt,
        "num_inference_steps": 2,
        "guidance_scale": 7.5
    }
    response = requests.post(url, json=payload, headers=headers)
    if response.status_code == 200:
        # Kết quả là base64, bạn cần giải mã
        import base64
        from PIL import Image
        from io import BytesIO

        output = response.json()
        b64_image = output.get("result", "")
        if b64_image:
            image_data = base64.b64decode(b64_image)
            return Image.open(BytesIO(image_data))
        else:
            raise Exception("No image result in Cloudflare response")
    else:
        raise Exception(f"Cloudflare API failed: {response.text}")

def generate_and_upload_image(prompt, model, post_id, wp_domain_url, wordpress_token):
    try:
        try:
            print("✨ Trying HuggingFace...")
            client = InferenceClient(
                provider="fal-ai",
                api_key=config.HF_API_KEY
            )
            image = client.text_to_image(prompt, model=model)
        except Exception as e:
            print(f"⚠️ HuggingFace failed: {str(e)}")
            print("✨ Falling back to Cloudflare...")
            image = generate_with_cloudflare(prompt)

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        temp_image_path = f"generated_image_{timestamp}.jpg"
        image.save(temp_image_path, format="JPEG")

        upload_result = cloudinary.uploader.upload(temp_image_path, folder="ai_generated_images")
        image_url = upload_result.get("secure_url")
        print(f"✅ Image uploaded: {image_url}")

        # Cập nhật ảnh đại diện
        wordpress_api_url = f"{wp_domain_url}/wp-json/fifu-api/v1/set-image"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {wordpress_token}"
        }
        data = {"post_id": post_id, "image_url": image_url}
        response = requests.post(wordpress_api_url, json=data, headers=headers)
        if response.status_code == 200:
            print("✅ Featured Image updated successfully!")
        else:
            print("❌ Failed to update Featured Image:", response.text)

        if os.path.exists(temp_image_path):
            os.remove(temp_image_path)
            print("🗑️ Temporary file deleted.")

        return image_url

    except Exception as e:
        print("❌ Error:", str(e))
        return None
    
def get_wordpress_token(wp_domain_url,username, password):
    url = wp_domain_url + "/wp-json/jwt-auth/v1/token"
    headers = {"Content-Type": "application/json"}
    data = {"username": username, "password": password}
    response = requests.post(url, json=data, headers=headers)
    
    if response.status_code == 200:
        json_response = response.json()
        # Kiểm tra xem key "data" và "token" có tồn tại không
        if "data" in json_response and "token" in json_response["data"]:
            token = json_response["data"]["token"]
            print("✅ Token retrieved successfully!")
            redis_client.setex(config.TOKEN_KEY, 3600, token)
            return token

    print("❌ Failed to get token:", response.text)
    return None

def is_token_valid(wp_domain_url,token):
    url = wp_domain_url+"/wp-json/jwt-auth/v1/token/validate"
    headers = {"Authorization": f"Bearer {token}"}
    response = requests.post(url, headers=headers)
    if response.status_code == 200:
        result = response.json()
        return result.get("success", False)
    return False

def get_valid_token(wp_domain_url,username, password):
    token = redis_client.get(config.TOKEN_KEY)
    if token:
        print("🔍 Checking token validity...")
        if is_token_valid(wp_domain_url,token):
            print("✅ Token is valid.")
            return token
        else:
            print("❌ Token is invalid or expired. Refreshing...")
    return get_wordpress_token(wp_domain_url,username, password)

def get_image_url(query):
    headers = {"Authorization": f"Client-ID {config.IMAGE_API_KEY}"}
    params = {"query": query, "per_page": 1}
    response = requests.get(config.IMAGE_BASE_URL, headers=headers, params=params)
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
        return None
    except HttpError as error:
        print(f"An error occurred: {error}")
        return None
    
def format_title(title):
    title = title.replace("**", "")
    title = re.sub(r"\(Word count:.*?\)", "", title).strip()
    return title

def format_content(content):
    phrases_to_remove = [
        "Here is the completed article based on the optimized outline:",
        "**Optimized Outline: Volvo's Electric Hybrid Efforts**",
        "I hope this meets your requirements! Let me know if you need any further modifications."
    ]
    content_lines = content.splitlines() if content is not None else []
    filtered_lines = [
        line for line in content_lines
        if not any(phrase in line for phrase in phrases_to_remove) and "Word count:" not in line
    ]
    processed_lines = []
    for line in filtered_lines:
        stripped_line = line.strip()
        stripped_line = re.sub(r"^Section\s+\d+:\s*", "", stripped_line)
        if stripped_line.startswith("**") and stripped_line.endswith("**"):
            h2_content = stripped_line.strip("**").strip()
            processed_lines.append(f"<h2>{h2_content}</h2>")
        else:
            processed_lines.append(stripped_line)
    formatted_content = "\n".join(processed_lines).strip()
    return formatted_content


def call_local_ollama(prompt, model="llama3:8b"):
    data = {"model": model, "prompt": prompt, "stream": False}
    try:
        response = requests.post(config.LOCAL_API_URL, json=data)
        if response.status_code == 200:
            response_data = response.json()
            return response_data.get("response", "No response found")
        else:
            print(f"Error: {response.status_code}, {response.text}")
            return None
    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}")
        return None
    
def call_ai_model(prompt, model="llama3:8b"):
    if config.USE_OPENROUTER_API:
        headers = {
            "Authorization": f"Bearer {config.OPENROUTER_API_KEY}",
            "Content-Type": "application/json"
        }
        data = {
            "model": config.MODEL_AI,
            "messages": [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
        }
        api_url = config.OPENROUTER_API_URL
    else:
        data = {"model": model, "prompt": prompt, "stream": False}
        api_url = config.LOCAL_API_URL
    try:
        response = requests.post(api_url, json=data, headers=headers if config.USE_OPENROUTER_API else None)
        if response.status_code == 200:
            response_data = response.json()
            return response_data.get("response", response_data.get("choices", [{}])[0].get("message", {}).get("content", "No response found"))
        else:
            print(f"Error: {response.status_code}, {response.text}")
            return None
    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}")
        return None
    
def confirm_seo_analytics():
    prompt = f"""
        You are a content strategist specializing in SEO. Your task is to refine content strategy to attract organic traffic, improve search engine rankings, and deeply resonate with the target audience.
        Please follow these guidelines while analyzing and enhancing the content:
        ●   Relevance: Ensure all content is directly relevant to the website's core topics, products, or services.
        ●   Depth & Value: Prioritize in-depth, informative content that genuinely addresses user needs and questions.
        ●   Search Intent Match: Carefully align the content with what users are actually searching for (e.g., informational articles, product comparisons, how-to guides).
        ●   Target Audience Understanding: Craft content that speaks directly to the interests, pain points, and language of the ideal customer.
        ●   Trends & Freshness: Incorporate trending topics or emerging industry news when relevant, while keeping content up-to-date.
        ●   Semantic Richness: Use a variety of related terms and phrases naturally throughout the content to capture a wider range of search queries.
        ●   Readability & Engagement: Write in a clear, engaging style that keeps readers on the page.
        Please present your findings in a structured format, including the following:
        ●   Content Area: Identify specific pages or sections of content.
        ●   Current Status: Briefly assess the content's strengths and weaknesses.
        ●   Suggested Improvements: Provide actionable recommendations for improving relevance, depth, search intent match, audience appeal, etc.
        ●   SEO Considerations: Highlight opportunities to optimize title tags, meta descriptions, header tags, and internal linking for relevant keywords.
        ●   Additional Notes: Any further insights on content promotion, link building, or other strategies to boost visibility.
        Feel free to use any relevant content analysis and SEO tools to aid your assessment. Remember, the goal is to create a content strategy that not only ranks well but truly resonates with readers and drives meaningful engagement.
        Do you understand these revised instructions? You must response YES or NO. If so, acknowledge and await my next prompt.
    """
    return call_ai_model(prompt)

def analyze_content(keyword):
    prompt = f"""
    You are a content strategist specializing in SEO. Create an outline for a blog targeting the keyword '{keyword}'. 
    Include an optimized title, meta description, H2 sections, and word count for each section.
    """
    return call_ai_model(prompt)


def find_secondary_keywords(keyword):
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
    result_text = call_ai_model(prompt)
    try:
        sections = {
            "Secondary Keywords": [],
            "NLP Keywords": [],
            "LSI Keywords": []
        }
        for section_name in sections.keys():
            pattern = rf"\*\*{section_name}\*\*.*?(?=(\*\*|$))"
            match = re.search(pattern, result_text, re.DOTALL)
            if match:
                content = match.group()
                keywords = re.findall(r"\d+\.\s+(.*)", content)
                sections[section_name] = keywords
        top_3_secondary_keywords = sections["Secondary Keywords"][:3]
        top_3_nlp_keywords = sections["NLP Keywords"][:3]
        top_3_lsi_keywords = sections["LSI Keywords"][:3]
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
        print(f"Error processing secondary keywords: {e}")
        return {
            "top_3_secondary_keywords": [],
            "top_3_nlp_keywords": [],
            "top_3_lsi_keywords": [],
            "concatenated_secondary_keywords": "",
            "concatenated_nlp_lsi_keywords": ""
        }
    

def find_longterm_keywords(keyword):
    prompt = f"""
    Identify long-tail keywords related to '{keyword}' that are relevant and have low competition but decent search volume.
    Arrange them in separate lists.
    """
    result = call_ai_model(prompt)
    all_keywords = []
    try:
        long_tail_keywords = result.get("long_tail_keywords", [])
        all_keywords = long_tail_keywords
        top_3_keywords = all_keywords[:3]
        concatenated_keywords = ", ".join(all_keywords)
        return {
            "top_3_longtail_keywords": top_3_keywords,
            "concatenated_keywords": concatenated_keywords
        }
    except Exception as e:
        print(f"Error processing long-tail keywords: {e}")
        return {
            "top_3_longtail_keywords": [],
            "concatenated_keywords": ""
        }
    
def analyze_site_keywords(keyword, site_list):
    results = {}
    all_keywords = []
    for site in site_list:
        prompt = f"""
        Analyze the top-ranking pages of {site} and identify the keywords they are targeting for organic search traffic. 
        Focus on keywords that are relevant to '{keyword}' and have high search volume and moderate to low competition.
        """
        analysis_result = call_ai_model(prompt)
        try:
            site_keywords = analysis_result.get("keywords", [])
            all_keywords.extend(site_keywords)
            top_3_keywords = site_keywords[:3]
            results[site] = {
                "top_3_keywords": top_3_keywords,
                "all_site_keywords": site_keywords
            }
        except Exception as e:
            print(f"Error processing keywords for site {site}: {e}")
            results[site] = {
                "top_3_keywords": [],
                "all_site_keywords": []
            }
    concatenated_keywords = ", ".join(all_keywords)
    return {
        "site_results": results,
        "concatenated_keywords": concatenated_keywords
    }

def find_trending_keywords_and_topics(keyword):
    prompt = f"""
    Identify trending keywords and topics in '{keyword}' that have experienced a significant increase in search volume or popularity in recent weeks or months. 
    Consider factors like news events, social media trends, seasonal changes, and emerging technologies.
    """
    result = call_ai_model(prompt)
    all_keywords = []
    try:
        trending_keywords = result.get("trending_keywords", [])
        trending_topics = result.get("trending_topics", [])
        all_keywords = trending_keywords + trending_topics
        top_3_keywords = all_keywords[:3]
        concatenated_keywords = ", ".join(all_keywords)
        return {
            "top_3_trending_keywords": top_3_keywords,
            "concatenated_keywords": concatenated_keywords
        }
    except Exception as e:
        print(f"Error processing trending keywords: {e}")
        return {
            "top_3_trending_keywords": [],
            "concatenated_keywords": ""
        }
    
def find_local_keywords_and_phrases(keyword, location="US"):
    prompt = f"""
    Identify local keywords and phrases that people in '{location}' are using to search for information like '{keyword}'. 
    Include keywords that mention the location, specific neighborhoods, landmarks, or nearby areas. 
    Also, consider keywords that reflect local search intent, such as "near me".
    """
    local_keywords_result = call_ai_model(prompt)
    try:
        local_keywords = local_keywords_result.get("local_keywords", [])
        top_3_keywords = local_keywords[:3]
        concatenated_keywords = ", ".join(local_keywords)
        return {
            "top_3_keywords": top_3_keywords,
            "concatenated_keywords": concatenated_keywords
        }
    except Exception as e:
        print(f"Error processing local keywords: {e}")
        return {
            "top_3_keywords": [],
            "concatenated_keywords": ""
        }
    
def create_seo_content_outline(keyword):
    prompt = f"""
    Use Google search to look up the top 5 results and create an SEO content outline for '{keyword}'. 
    For each section, include the amount of words. 
    Also, include an optimized title, meta description, and H2s.
    """
    seo_outline = call_ai_model(prompt)
    return seo_outline

def optimize_outline(outline, keyword):
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
        title_start = result_text.find("**Title:**")
        title_end = result_text.find("\n", title_start)
        title = result_text[title_start + 10:title_end].strip() if title_start != -1 else ""
        meta_start = result_text.find("**Meta Description:**")
        meta_end = result_text.find("\n", meta_start)
        meta_description = (result_text[meta_start + 20:meta_end].strip() if meta_start != -1 else "")
        outline_start = result_text.find("**Optimized Outline:**")
        optimized_outline = (result_text[outline_start:].strip() if outline_start != -1 else result_text)
        return {
            "title": title,
            "meta_description": meta_description,
            "optimized_outline": optimized_outline,
        }
    except Exception as e:
        print(f"Error processing outline: {e}")
        return {
            "title": "",
            "meta_description": "",
            "optimized_outline": outline,
        }
    
def write_content(outline, keyword, secondaryKeywords, LSIandNLPKeywords):
    prompt = f"""
    Start writing the content with {outline}, one section at a time, auto next section and combine all section to an completed article. Utilize the {keyword}, secondary keywords: {secondaryKeywords}, LSI and NLP keywords : {LSIandNLPKeywords} listed through out the content naturally. Also, maintain the word count specified for each section. Note that I want the content to be written like it was written by a subject matter expert, without fluff or jargon. The content you produce should also have low AI content detection scores and should reflect the same when passed through AI content detectors. Write in a [tone/style: informative] voice that engages the reader. The content should sound like it was written by a person, not a machine. Avoid clichés, jargon, and overly complex sentence structures.. Focus on originality. Do not plagiarize existing work. Ensure the facts and ideas presented are accurate and well-researched.
    output is only completed article to retrieve data easily to post wordpress.
    """
    return call_ai_model(prompt)
