

from langchain.document_loaders import SeleniumURLLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import DeepLake
from langchain import PromptTemplate
from langchain.llms import OpenAI
from langchain import LLMChain

# we'll use information from the following articles
urls = ['https://beebom.com/what-is-nft-explained/',
        'https://beebom.com/how-delete-spotify-account/',
        'https://beebom.com/how-download-gif-twitter/',
        'https://beebom.com/how-use-chatgpt-linux-terminal/',
        'https://beebom.com/how-delete-spotify-account/',
        'https://beebom.com/how-save-instagram-story-with-music/',
        'https://beebom.com/how-install-pip-windows/',
        'https://beebom.com/how-check-disk-usage-linux/']

# 1: Load the data from urls
loader = SeleniumURLLoader(urls=urls)
entire_data = loader.load()

# 2: Split them into chunks using the CharacterTextSplitter with a chunk size of 1000 and no overlap:
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
data = text_splitter.split_documents(entire_data)

# 3: Compute the embeddings using OpenAIEmbeddings
open_ai_token="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"
embeddings = OpenAIEmbeddings(model="text-embedding-ada-002", openai_api_key=open_ai_token)

# 4: store them in a Deep Lake vector store on the cloud
my_activeloop_org_id = "srishtysuman2919" 
customer_support_chatbot = "customer_support_chatbot"
dataset_path = f"https:/app.activeloop.ai/srishtysuman2919/{customer_support_chatbot}"
deep_lake_token="eyJhbGciOiJIUzUxMiIsImlhdCI6MTcwMzk1NDQzMCwiZXhwIjoxNzM1NTc2ODE2fQ.eyJpZCI6InNyaXNodHlzdW1hbjI5MTkifQ.j_B5pt-7UL48fhp5cXStId-DNydAiuqDV3KlUC9cgE_ivDRyPaCmNnrWyTKp7Qkjx_rFL3M_T1yUR9HhG5vxmA"
db = DeepLake(dataset_path=dataset_path, embedding_function=embeddings, token=deep_lake_token)
db.add_documents(data)

# 5: To retrieve the most similar chunks to a given query, we can use the similarity_search method of the Deep Lake vector store
query = "how to check disk usage in linux?"
docs = db.similarity_search(query)
print(docs[0].page_content)

# 6: Craft a prompt for GPT-3-> create a prompt template that incorporates role-prompting, relevant Knowledge Base information, and the user's question:
# let's write a prompt for a customer support chatbot that
# answer questions using information extracted from our db
template = """You are an exceptional customer support chatbot that gently answer questions.

You know the following context information.

{chunks_formatted}

Answer to the following question from a customer. Use only information from the previous context information. Do not invent stuff.

Question: {query}

Answer:"""

prompt = PromptTemplate(
    input_variables=["chunks_formatted", "query"],
    template=template,
)

# 7: Utilize the GPT3 model with a temperature of 0 for text generation
llm = OpenAI(model="text-davinci-003", temperature=0, openai_api_key="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")

# user question
query = "How to check disk usage in linux?"

# 8: retrieve relevant chunks
docs = db.similarity_search(query)
retrieved_chunks = [doc.page_content for doc in docs]

# 9: format the prompt
chunks_formatted = "\n\n".join(retrieved_chunks)
prompt_formatted = prompt.format(chunks_formatted=chunks_formatted, query=query)

# 10: generate answer
answer = llm(prompt_formatted)
print(answer)