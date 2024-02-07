from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import DeepLake
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA

# create our documents
texts = [
    "Napoleon Bonaparte was born in 15 August 1769",
    "Louis XIV was born in 5 September 1638",
    "Lady Gaga was born in 28 March 1986",
    "Michael Jeffrey Jordan was born in 17 February 1963"
]
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
docs = text_splitter.create_documents(texts)

# initialize embeddings model
embeddings = OpenAIEmbeddings(model="text-embedding-ada-002", openai_api_key="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")

# create Deep Lake dataset
# TODO: use your organization id here. (by default, org id is your username)
my_activeloop_org_id =  "srishtysuman2919" 
my_activeloop_dataset_name = "langchain_course_embeddings"
dataset_path = f"https:/app.activeloop.ai/srishtysuman2919/{my_activeloop_dataset_name}"
deeplake_api_token="eyJhbGciOiJIUzUxMiIsImlhdCI6MTcwMzk1NDQzMCwiZXhwIjoxNzM1NTc2ODE2fQ.eyJpZCI6InNyaXNodHlzdW1hbjI5MTkifQ.j_B5pt-7UL48fhp5cXStId-DNydAiuqDV3KlUC9cgE_ivDRyPaCmNnrWyTKp7Qkjx_rFL3M_T1yUR9HhG5vxmA"
db = DeepLake(dataset_path=dataset_path, embedding_function=embeddings, token=deeplake_api_token)

# add documents to our Deep Lake dataset
db.add_documents(docs)

# create retriever from db
retriever = db.as_retriever()

# istantiate the llm wrapper
model = ChatOpenAI(model='gpt-3.5-turbo', openai_api_key="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")

# create the question-answering chain
qa_chain = RetrievalQA.from_llm(model, retriever=retriever)

# ask a question to the chain
qa_chain.run("When was Michael Jordan born?")

