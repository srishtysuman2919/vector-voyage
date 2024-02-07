'''
    Deep Lake provides storage for embeddings and their corresponding metadata in the context of LLM apps. 
    It enables hybrid searches on these embeddings and their attributes for efficient data retrieval. 
    It also integrates with LangChain, facilitating the development and deployment of applications.

    Deep Lake provides several advantages over the typical vector store:
    1. It’s multimodal, which means that it can be used to store items of diverse modalities, such as texts, images, audio, and video, along with their vector representations. 
    2. It’s serverless, which means that we can create and manage cloud datasets without creating and managing a database instance. This aspect gives a great speedup to new projects.
    3. Last, it’s possible to easily create a data loader out of the data loaded into a Deep Lake dataset. 
       It is convenient for fine-tuning machine learning models using common frameworks like PyTorch and TensorFlow.

'''

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import DeepLake
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA

from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentType

# Before executing the following code, make sure to have your
# Activeloop key saved in the “ACTIVELOOP_TOKEN” environment variable.

# instantiate the LLM and embeddings models
open_ai_token="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"
llm = OpenAI(model="text-davinci-003", temperature=0, openai_api_key=open_ai_token)
embeddings = OpenAIEmbeddings(model="text-embedding-ada-002", openai_api_key=open_ai_token)

# create our documents
texts = [
    "Napoleon Bonaparte was born in 15 August 1769",
    "Louis XIV was born in 5 September 1638"
]
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
docs = text_splitter.create_documents(texts)

# create Deep Lake dataset
# TODO: use your organization id here. (by default, org id is your username)
my_activeloop_org_id = "srishtysuman2919" 
my_activeloop_dataset_name = "langchain_course_from_zero_to_hero"
dataset_path = "https:/app.activeloop.ai/srishtysuman2919/langchain_course_from_zero_to_hero"
deep_lake_token="eyJhbGciOiJIUzUxMiIsImlhdCI6MTcwMzk1NDQzMCwiZXhwIjoxNzM1NTc2ODE2fQ.eyJpZCI6InNyaXNodHlzdW1hbjI5MTkifQ.j_B5pt-7UL48fhp5cXStId-DNydAiuqDV3KlUC9cgE_ivDRyPaCmNnrWyTKp7Qkjx_rFL3M_T1yUR9HhG5vxmA"
db = DeepLake(dataset_path=dataset_path, embedding_function=embeddings, token=deep_lake_token)

# # add documents to our Deep Lake dataset
db.add_documents(docs)


# let's create a RetrievalQA chain
retrieval_qa = RetrievalQA.from_chain_type(
	llm=llm,
	chain_type="stuff",
	retriever=db.as_retriever()
)

# Next, let's create an agent that uses the RetrievalQA chain as a tool
tools = [
    Tool(
        name="Retrieval QA System",
        func=retrieval_qa.run,
        description="Useful for answering questions."
    ),
]

agent = initialize_agent(
	tools,
	llm,
	agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
	verbose=True,
    max_iterations=6
)

# Finally, we can use the agent to ask a question:
response = agent.run("When was Napoleone born?")
print(response)

# We should see something similar to the following printed output. 
# Here, the agent used the “Retrieval QA System” tool with the query “When was Napoleone born?” 
# which is then run on our new Deep Lake dataset, returning the most similar document 
# (i.e., the document containing the date of birth of Napoleon). This document is eventually used to generate the final output.



# Let’s add an example of reloading an existing vector store and adding more data
# load the existing Deep Lake dataset and specify the embedding function
db = DeepLake(dataset_path=dataset_path, embedding_function=embeddings, token=deep_lake_token)

# create new documents
texts = [
    "Lady Gaga was born in 28 March 1986",
    "Michael Jeffrey Jordan was born in 17 February 1963"
]
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
docs = text_splitter.create_documents(texts)

# add documents to our Deep Lake dataset
db.add_documents(docs)

# We then recreate our previous agent and ask a question that can be answered only by the last documents added
# instantiate the wrapper class for GPT3
llm = OpenAI(model="text-davinci-003", temperature=0, openai_api_key=open_ai_token)

# create a retriever from the db
retrieval_qa = RetrievalQA.from_chain_type(
	llm=llm, chain_type="stuff", retriever=db.as_retriever()
)

# instantiate a tool that uses the retriever
tools = [
    Tool(
        name="Retrieval QA System",
        func=retrieval_qa.run,
        description="Useful for answering questions."
    ),
]

# create an agent that uses the tool
agent = initialize_agent(
	tools,
	llm,
	agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
	verbose=True
)

# Let’s now test our agent with a new question
response = agent.run("When was Michael Jordan born?")
print(response)

# The LLM successfully retrieves accurate information by using the power of Deep Lake as a vector store and the OpenAI language model.