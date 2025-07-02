import requests
from thirdparty.redisconnection import redis_client
import config
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from huggingface_hub import InferenceClient
import re
import datetime
import os
import cloudinary
import cloudinary.uploader
from PIL import Image
from io import BytesIO
import logging
import time

# -------------------------
# Các hàm hỗ trợ (helper functions)
# -------------------------

# Cấu hình Cloudinary
cloudinary.config(
    cloud_name=config.CLOUDINARY_CLOUD_NAME,
    api_key=config.CLOUDINARY_API_KEY,
    api_secret=config.CLOUDINARY_API_SECRET
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def generate_with_cloudflare(prompt, num_inference_steps=30, guidance_scale=9.0, retries=3):
    url = "https://api.cloudflare.com/client/v4/accounts/eb9da718bb875b75af5a984d0cd76fe7/ai/run/@cf/stabilityai/stable-diffusion-xl-base-1.0"
    headers = {
        "Authorization": f"Bearer {config.CLOUDFLARE_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "prompt": prompt,
        "num_inference_steps": num_inference_steps,
        "guidance_scale": guidance_scale
    }

    for attempt in range(retries):
        try:
            logger.info(f"Attempt {attempt + 1}/{retries} - Sending request to {url} with prompt: {prompt}")
            response = requests.post(url, json=payload, headers=headers, timeout=60)
            logger.info(f"Response status: {response.status_code}")
            logger.info(f"Response headers: {response.headers}")
            logger.info(f"Response content type: {response.headers.get('content-type')}")
            logger.info(f"Response content length: {response.headers.get('content-length')} bytes")

            if response.status_code == 200:
                if response.headers.get('content-type') == 'image/png':
                    try:
                        image = Image.open(BytesIO(response.content))
                        logger.info("Image generated successfully from binary PNG data")
                        # Tối ưu kích thước (giảm xuống 800x800 để upload dễ hơn)
                        image = image.resize((738, 521), Image.Resampling.LANCZOS)
                        return image
                    except Exception as e:
                        logger.error(f"Image processing error: {e}")
                        raise Exception("Failed to process PNG image")
                else:
                    logger.error(f"Unexpected content type: {response.headers.get('content-type')}")
                    raise Exception("Unexpected response content type")
            else:
                logger.error(f"Cloudflare API failed: {response.status_code}, {response.text}")
                if attempt < retries - 1:
                    logger.info(f"Retrying due to API failure...")
                    time.sleep(2)
                    continue
                raise Exception(f"Cloudflare API failed: {response.status_code}, {response.text}")

        except requests.exceptions.RequestException as e:
            logger.error(f"Request failed: {e}")
            if attempt < retries - 1:
                logger.info(f"Retrying due to network error...")
                time.sleep(2)
                continue
            raise Exception(f"Network error: {e}")

    raise Exception("Failed to generate image after all retries")

def generate_and_upload_image(prompt, model, post_id, wp_domain_url, wordpress_token):
    try:
        try:
            print("✨ Trying HuggingFace...")
            client = InferenceClient(
                provider="hf-inference",
                api_key=config.HF_API_KEY
            )
            image = client.text_to_image(prompt, model=model)
            image = image.resize((738, 521), Image.Resampling.LANCZOS)
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
    youtube = build("youtube", "v3", developerKey=youtube_api_key, cache_discovery=False)
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
    
def call_ai_model(prompt, model="llama3:8b", provider=config.PROVIDER):
    if provider == "openrouter":
        headers = {
            "Authorization": f"Bearer {config.OPENROUTER_API_KEY}",
            "Content-Type": "application/json"
        }
        data = {
            "model": config.MODEL_AI,  # ví dụ: "mistralai/mistral-7b-instruct"
            "messages": [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
        }
        api_url = config.OPENROUTER_API_URL

    elif provider == "groq":
        headers = {
            "Authorization": f"Bearer {config.GROQ_API_KEY}",
            "Content-Type": "application/json"
        }
        data = {
            "model": config.GROQ_MODEL_AI,  # ví dụ: "llama3-8b-8192" hoặc "llama3-70b-8192"
            "messages": [{"role": "user", "content": prompt}]
        }
        api_url = config.GROQ_API_URL  # thường là https://api.groq.com/openai/v1/chat/completions

    else:  # local fallback
        data = {"model": model, "prompt": prompt, "stream": False}
        headers = None
        api_url = config.LOCAL_API_URL

    try:
        response = requests.post(api_url, json=data, headers=headers)
        if response.status_code == 200:
            response_data = response.json()
            # Groq/OpenRouter đều dùng OpenAI schema → có 'choices'
            return response_data.get("choices", [{}])[0].get("message", {}).get("content", "No response found")
        else:
            print(f"[{provider}] Error: {response.status_code}, {response.text}")
            return None
    except requests.exceptions.RequestException as e:
        print(f"[{provider}] Request failed: {e}")
        return None
    
def confirm_seo_analytics():
    prompt = f"""
        You are a content strategist specializing in SEO. Your task is to refine content strategy to attract organic traffic, improve search engine rankings, and deeply resonate with the target audience. Follow these guidelines while analyzing and enhancing the content:
        - Relevance: Ensure all content is directly relevant to the website's core topics, products, or services.
        - Depth & Value: Prioritize in-depth, informative content that addresses user needs and questions.
        - Search Intent Match: Align content with user search intent (e.g., informational articles, product comparisons, how-to guides).
        - Target Audience Understanding: Craft content for the interests, pain points, and language of the ideal customer (e.g., pet owners or EV enthusiasts in the USA).
        - Trends & Freshness: Incorporate trending topics or industry news when relevant, keeping content up-to-date.
        - Semantic Richness: Use related terms and phrases naturally to capture a wider range of search queries.
        - Readability & Engagement: Write in simple English, targeting Grade 8 readability, avoiding passive voice and complex words (e.g., use 'use' instead of 'utilize'). Ensure an engaging style to keep readers on the page.
        - E-E-A-T: Include stats or examples from credible sources (e.g., IEA or Electrek for EVs, PetMD for pets). Add real-world examples (e.g., 'As a pet owner, I tested...'). Include an author bio (e.g., 'Written by [Name], an enthusiast in [topic]') to enhance trustworthiness.
        Present your findings in a structured format:
        - Content Area: Identify specific pages or sections.
        - Current Status: Assess strengths and weaknesses.
        - Suggested Improvements: Provide actionable recommendations for relevance, depth, search intent, and audience appeal.
        - SEO Considerations: Optimize title tags (55-60 characters), meta descriptions (150-160 characters), header tags, and internal links (2-3 per article) for relevant keywords.
        - Additional Notes: Include insights on content promotion, link building, or strategies to boost visibility.
        Do you understand these instructions? Respond YES or NO and await my next prompt.
    """
    return call_ai_model(prompt)

def analyze_content(keyword):
    prompt = f"""
        You are a content strategist specializing in SEO. Create an SEO content outline for a blog targeting the keyword '{keyword}'. Follow these guidelines:
        - Search Intent: Align the outline with user search intent (e.g., informational, commercial investigation, how-to guide).
        - Structure: Include an optimized title (55-60 characters), meta description (150-160 characters), and 4-5 H2 sections with suggested word counts for each.
        - Readability: Ensure the outline supports content written in simple English, targeting Grade 8 readability, avoiding passive voice and complex words.
        - E-E-A-T: Suggest including stats or examples from credible sources (e.g., IEA or Electrek for EVs, PetMD for pets) and real-world examples relevant to '{keyword}' (e.g., 'As a user, I tested...').
        - SEO: Incorporate the primary keyword '{keyword}' at 1-2% density and use related LSI/NLP keywords naturally. Suggest 2-3 internal links to related articles and 3-5 images with alt text containing '{keyword}' or related terms.
        - Analyze the top 5 Google search results for '{keyword}' to ensure the outline is competitive and relevant.
        Output format:
        - Title: [Optimized title]
        - Meta Description: [Optimized meta description]
        - Outline:
        - H2: [Section title] ([Word count])
            - Key points: [Brief description]
        - H2: [Section title] ([Word count])
            - Key points: [Brief description]
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
        You are a content strategist specializing in SEO. Create an SEO content outline for '{keyword}' by analyzing the top 5 Google search results. Follow these guidelines:
        - Search Intent: Align the outline with user search intent (e.g., informational, commercial investigation, how-to guide).
        - Structure: Include an optimized title (55-60 characters), meta description (150-160 characters), and 4-5 H2 sections with word counts for each.
        - Readability: Ensure the outline supports content written in simple English, targeting Grade 8 readability, avoiding passive voice and complex words.
        - E-E-A-T: Suggest including stats or examples from credible sources (e.g., IEA or Electrek for EVs, PetMD for pets) and real-world examples relevant to '{keyword}'.
        - SEO: Incorporate '{keyword}' at 1-2% density, use related LSI/NLP keywords naturally, and suggest 2-3 internal links and 3-5 images with alt text containing '{keyword}' or related terms.
        - Important: All examples (e.g., 'pet-friendly electric cars 2025') are illustrative. Replace them with content related to '{keyword}'.
        Output format:
        - Title: [Optimized title]
        - Meta Description: [Optimized meta description]
        - Outline:
        - H2: [Section title] ([Word count])
            - Key points: [Brief description]
        - H2: [Section title] ([Word count])
            - Key points: [Brief description]
        """
    return call_ai_model(prompt)

def optimize_outline(outline, keyword):
    prompt = f"""
        Please optimize this outline to target the primary keyword '{keyword}' at 1-2% density. Use related LSI/NLP keywords naturally. Add an FAQ section to the outline and provide an optimized title and meta description. Follow these guidelines:
        - Readability: Ensure the outline supports content written in simple English, targeting Grade 8 readability, avoiding passive voice and complex words.
        - E-E-A-T: Suggest including stats or examples from credible sources (e.g., IEA or Electrek for EVs, PetMD for pets) and real-world examples relevant to '{keyword}' in the FAQ section.
        - SEO: Suggest 2-3 internal links and 3-5 images with alt text containing '{keyword}' or related terms. Ensure the title is 55-60 characters and meta description is 150-160 characters.
        - Important: All examples (e.g., 'pet-friendly electric cars 2025') are illustrative. Replace them with content related to '{keyword}'.
        - Output format:
            **Title:** [Optimized title]
            **Meta Description:** [Optimized meta description]
            **Optimized Outline:** [Optimized outline with FAQ section]
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
        Write a complete article based on {outline}, one section at a time, combining all sections into a cohesive piece. Utilize the primary keyword '{keyword}' at 1-2% density, secondary keywords: {secondaryKeywords}, and LSI/NLP keywords: {LSIandNLPKeywords} naturally throughout the content. Target a word count of 1000-1500 words. Follow these guidelines:
        - Tone/style: Informative, engaging, written like a subject matter expert, without fluff or jargon.
        - Readability: Target Grade 8 readability, using simple English, avoiding passive voice and complex words (e.g., use 'use' instead of 'utilize').
        - E-E-A-T:
        - Include stats or examples from credible sources relevant to '{keyword}' (e.g., IEA, Electrek, Hyundai owner’s manual).
        - Add 2-3 real-world examples from EV owners or experts relevant to '{keyword}'.
        - Include an author bio relevant to '{keyword}' at the end.
        - SEO:
        - Include a meta title (55-60 characters) and meta description (150-160 characters) incorporating '{keyword}'.
        - Suggest 3 image placements with alt text incorporating '{keyword}' or related terms (e.g., '{keyword} 2025' or a variation).
        - Include 2-3 internal links to related articles relevant to '{keyword}'.
        - Formatting:
        - Use HTML tags for headings (e.g., <h2>Section Title</h2>) instead of Markdown.
        - Format the primary keyword '{keyword}' using <strong><em>{keyword}</em></strong> at least once per section.
        - Ensure pure HTML output, free of Markdown symbols.
        - Insert exactly three placeholders: [INSERT_IMAGE_1] after Introduction, [INSERT_IMAGE_2] after Common Symbols, and [INSERT_IMAGE_3] after Conclusion.
        - Structure and Depth:
        - Expand the introduction with EV market trends and importance of '{keyword}'.
        - Add 'Advanced Dashboard Features' covering unique technologies.
        - Include 'Real Owner Stories' with 2-3 personal experiences.
        - Expand FAQ with 5-7 questions and detailed answers.
        - Image Integration:
        - Suggest 3 image placements within sections: one after Introduction, one after Common Symbols, and one after Conclusion.
        - Provide descriptive prompts for images based on '{keyword}' (e.g., 'dashboard with {keyword} in daylight', 'close-up of {keyword}').
        - Use placeholders [INSERT_IMAGE_1], [INSERT_IMAGE_2], and [INSERT_IMAGE_3] for image insertion.
        - Originality: Ensure low AI detection scores by varying sentence structures and unique examples. Avoid plagiarism.
        - Important: Replace illustrative examples (e.g., 'pet-friendly electric cars 2025') with '{keyword}' or related terms.
        - Output: Complete article in HTML, including meta title, meta description, and author bio, with image placeholders or URLs.
        """
    content = call_ai_model(prompt, model="meta-llama/llama-3.1-8b-instruct:free")
    
    if content:
        image_positions = {
            "[INSERT_IMAGE_1]": f"dashboard with {keyword} in daylight",
            "[INSERT_IMAGE_2]": f"close-up of {keyword}",
            "[INSERT_IMAGE_3]": f"interior with {keyword} 2025"
        }
        for pos, image_prompt in image_positions.items():
            if pos in content:  # Kiểm tra xem placeholder có trong content không
                try:
                    # Tạo ảnh bằng Cloudflare
                    image = generate_with_cloudflare(image_prompt)
                    # Upload lên Cloudinary và lấy URL
                    image_url = upload_to_cloudinary(image)
                    alt_text = image_prompt.replace("'", "").replace('"', '')
                    content = content.replace(pos, f'<img src="{image_url}" alt="{alt_text}">')
                except Exception as e:
                    logger.warning(f"Image creation/upload failed at {pos}: {e} - Skipping image")
                    content = content.replace(pos, "")  # Loại bỏ placeholder nếu lỗi
            else:
                logger.warning(f"Placeholder {pos} not found in content - Skipping image insertion")
    return content

def upload_to_cloudinary(image):
    try:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        temp_image_path = f"generated_image_{timestamp}.jpg"
        image.save(temp_image_path, format="JPEG")

        upload_result = cloudinary.uploader.upload(temp_image_path, folder="ai_generated_images")
        logger.info(f"Upload successful: {upload_result['secure_url']}")

        if os.path.exists(temp_image_path):
            os.remove(temp_image_path)
            print("🗑️ Temporary internal content file deleted.")
        return upload_result["secure_url"]
    except Exception as e:
        logger.error(f"Cloudinary upload failed: {e}")
        raise Exception(f"Cloudinary upload failed: {e}")