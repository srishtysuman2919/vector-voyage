'''
    Workflow for building a news article summarizer:
    1. Install required libraries: To get started, ensure you have the necessary libraries installed: requests, newspaper3k, and langchain.
    2. Scrape articles: Use the requests library to scrape the content of the target news articles from their respective URLs.
    3. Extract titles and text: Employ the newspaper library to parse the scraped HTML and extract the titles and text of the articles.
    4. Preprocess the text: Clean and preprocess the extracted texts to make them suitable for input to ChatGPT.
    5. Generate summaries: Utilize ChatGPT to summarize the extracted articles' text concisely.
    6. Output the results: Present the summaries along with the original titles, allowing users to grasp the main points of each article quickly.
'''

# The following code fetches articles from a list of URLs using the requests library with a custom User-Agent header. 
# It then extracts the title and text of each article using the newspaper library
import requests
from newspaper import Article
from langchain.schema import HumanMessage

headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/89.0.4389.82 Safari/537.36'
}

article_url = "https://www.artificialintelligence-news.com/2022/01/25/meta-claims-new-ai-supercomputer-will-set-records/"

session = requests.Session()

try:
    response = session.get(article_url, headers=headers, timeout=10)
    
    if response.status_code == 200:
        article = Article(article_url)
        article.download()
        article.parse()
        
        print(f"Title: {article.title}")
        print(f"Text: {article.text}")
        
    else:
        print(f"Failed to fetch article at {article_url}")
except Exception as e:
    print(f"Error occurred while fetching article at {article_url}: {e}")

# The next code imports essential classes and functions from the LangChain and sets up a ChatOpenAI instance with a temperature of 0 for controlled response generation. 
# Additionally, it imports chat-related message schema classes, which enable the smooth handling of chat-based tasks. 
# The following code will start by setting the prompt and filling it with the article’s content.

# we get the article data from the scraping part
article_title = article.title
article_text = article.text

# prepare template for prompt
template = """You are a very good assistant that summarizes online articles.
Here's the article you want to summarize.
==================
Title: {article_title}
{article_text}
==================
Write a summary of the previous article.
"""

prompt = template.format(article_title=article.title, article_text=article.text)

messages = [HumanMessage(content=prompt)]

# The HumanMessage is a structured data format representing user messages within the chat-based interaction framework. 
# The ChatOpenAI class is utilized to interact with the AI model, while the HumanMessage schema provides a standardized representation of user messages. 
# The template consists of placeholders for the article's title and content, which will be substituted with the actual article_title and article_text. 
# This process simplifies and streamlines the creation of dynamic prompts by allowing you to define a template with placeholders and then replace them with actual data when needed.


from langchain.chat_models import ChatOpenAI
# load the model
chat = ChatOpenAI(model_name="gpt-4", temperature=0, openai_api_key="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")

# As we loaded the model and set the temperature to 0. We’d use the chat() instance to generate a summary by passing a single HumanMessage object containing the formatted prompt. 
# The AI model processes this prompt and returns a concise summary:
# generate summary
summary = chat(messages)
print(summary.content)

# If we want a bulleted list, we can modify a prompt and get the result.
# prepare template for prompt
template = """You are an advanced AI assistant that summarizes online articles into bulleted lists.
Here's the article you need to summarize.
==================
Title: {article_title}
{article_text}
==================
Now, provide a summarized version of the article in a bulleted list format.
"""

# format prompt
prompt = template.format(article_title=article.title, article_text=article.text)

# generate summary
summary = chat([HumanMessage(content=prompt)])
print(summary.content)

# If you want to get the summary in French, you can instruct the model to generate the summary in French language.
# However, please note that GPT-4's main training language is English and while it has a multilingual capability, the quality may vary for languages other than English.
# Here's how you can modify the prompt.
# prepare template for prompt
template = """You are an advanced AI assistant that summarizes online articles into bulleted lists in French.
Here's the article you need to summarize.
==================
Title: {article_title}
{article_text}
==================
Now, provide a summarized version of the article in a bulleted list format, in French.
"""

# format prompt
prompt = template.format(article_title=article.title, article_text=article.text)

# generate summary
summary = chat([HumanMessage(content=prompt)])
print(summary.content)


'''
    Summarizing news article in French:
    1. First, we obtain the article data, including the title and text. 
    
    2. We then prepare a template for the prompt we want to give to the AI model. 
    This prompt is designed to simulate a conversation with the model, telling it that it's an "advanced AI assistant" and giving it a specific task - 
    to summarize the article into a bulleted list in French.

    3. Once the template is ready, we load the GPT-4 model using ChatOpenAI class with a certain temperature setting, which influences the randomness of the model's outputs. 
    We then format the prompt using the article data.

    4. The core part of the process is when we pass the formatted prompt to the model. The model parses the prompt, understands the task, and generates a summary accordingly. 
    It uses its vast knowledge, trained on diverse internet text, to comprehend and summarize the article in French.

    5. Lastly, the generated summary, which is a response from the model, is printed. 
    The summary is expected to be a concise, bullet-point version of the article in French, just as we instructed the model in the prompt.

    6. In essence, we are guiding the model using natural language instructions to generate the desired output. T
    his interaction is akin to how we might ask a human assistant to perform a task, making it a powerful and intuitive solution for a variety of applications.
'''
