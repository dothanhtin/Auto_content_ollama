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
    prompt = f"""
    Identify secondary, NLP, and LSI keywords related to '{keyword}' that are relevant and have low competition but decent search volume.
    Arrange them in separate lists.
    """
    return call_local_ollama(prompt)

# 3. Tối ưu outline
def optimize_outline(outline, keyword):
    prompt = f"""
    Optimize this outline to target the primary keyword '{keyword}' at a 3% density. Use other keywords naturally. Add an FAQ section.
    Outline: {outline}
    """
    return call_local_ollama(prompt)

# 4. Viết nội dung
def write_content(outline, keyword):
    prompt = f"""
    Write a blog based on the following outline. Use the keyword '{keyword}' at a 3% density. The content should be engaging, informative, and optimized for SEO.
    Outline: {outline}
    """
    return call_local_ollama(prompt)

# Quy trình thực hiện
if __name__ == "__main__":
    # Từ khóa chính
    main_keyword = "Electric cars with advanced AI parking systems"

    # Bước 0: Xác định vai trò SEO analytics cho AI
    confirm = confirm_seo_analytics()

    if("YES" in confirm):
        # Bước 1: Phân tích và tạo outline
        outline = analyze_content(main_keyword)
        print("Outline:\n", outline)

        # Bước 2: Tìm từ khóa phụ
        secondary_keywords = find_secondary_keywords(main_keyword)
        print("\nSecondary Keywords:\n", secondary_keywords)

        # Bước 3: Tối ưu outline
        optimized_outline = optimize_outline(outline, main_keyword)
        print("\nOptimized Outline:\n", optimized_outline)

        # Bước 4: Viết nội dung
        content = write_content(optimized_outline, main_keyword)
        print("\nContent:\n", content)
    else:
        print("No SEO analytics!")
