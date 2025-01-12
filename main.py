import requests

# Cấu hình API cục bộ của Ollama
LOCAL_API_URL = "http://localhost:11434/api/generate"  # Endpoint API cục bộ

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
        Do you understand these revised instructions? You can response YES or NO. If so, acknowledge and await my next prompt.
    """
    return call_local_ollama(prompt)

# 1. Phân tích chiến lược nội dung
def analyze_content(keyword):
    prompt = f"""
    You are a content strategist specializing in SEO. Create an outline for a blog targeting the keyword '{keyword}'. 
    Include an optimized title, meta description, H2 sections, and word count for each section.
    """
    return call_local_ollama(prompt)

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
    Arrange them in separate lists.
    """
    # Gọi hàm xử lý của mô hình AI
    result_text = call_local_ollama(prompt)

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
    result = call_local_ollama(prompt)

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
        analysis_result = call_local_ollama(prompt)

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
    result = call_local_ollama(prompt)

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
    local_keywords_result = call_local_ollama(prompt)

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
    # Gọi hàm xử lý của mô hình (giả sử bạn có hàm `call_local_ollama` để giao tiếp với mô hình AI)
    seo_outline = call_local_ollama(prompt)
    return seo_outline


# 8. Tối ưu outline
def optimize_outline(outline, keyword):
    prompt = f"""
    Please Optimize this outline to target the primary keyword '{keyword}' at a 3% density. Use other keywords naturally. Add an FAQ section.
    Outline: {outline}
    """
    return call_local_ollama(prompt)

# 4. Viết nội dung
def write_content(outline, keyword, secondaryKeywords,LSIandNLPKeywords):
    prompt = f"""
    Start writing the content with {outline}. Utilize the {keyword}, secondary keywords: {secondaryKeywords}, LSI and NLP keywords : {LSIandNLPKeywords} listed through out the content naturally. Also, maintain the word count specified for each section. Note that I want the content to be written like it was written by a subject matter expert, without fluff or jargon. The content you produce should also have low AI content detection scores and should reflect the same when passed through AI content detectors. Write in a [tone/style: informative] voice that engages the reader. The content should sound like it was written by a person, not a machine. Avoid clichés, jargon, and overly complex sentence structures.. Focus on originality. Do not plagiarize existing work. Ensure the facts and ideas presented are accurate and well-researched.
    """
    return call_local_ollama(prompt)

# Quy trình thực hiện
if __name__ == "__main__":
    # Từ khóa chính
    main_keyword = "volvo cars electric hybrid"

    # Bước 0: Xác định vai trò SEO analytics cho AI
    confirm = confirm_seo_analytics()

    if("YES" in confirm):
        # Bước 1: Phân tích và tạo outline
        outline = create_seo_content_outline(main_keyword)

        # Bước 2: Tìm từ khóa phụ, nsi, npl keywords
        secondary_keywords = find_secondary_keywords(main_keyword)

        # Bước 3: Tối ưu outline
        optimized_outline = optimize_outline(outline, main_keyword)
        print("\nOptimized Outline:\n", optimized_outline)

        # Bước 4: Viết nội dung
        content = write_content(optimized_outline, main_keyword,secondary_keywords["concatenated_secondary_keywords"],secondary_keywords["concatenated_nlp_lsi_keywords"])
        print("\nContent:\n", content)
    else:
        print("No SEO analytics!")
