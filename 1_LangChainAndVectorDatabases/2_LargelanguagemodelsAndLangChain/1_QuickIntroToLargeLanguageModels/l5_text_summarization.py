'''
    Using LangChain, we can create a chain for text summarization.
'''
from langchain.chat_models import ChatOpenAI
from langchain import PromptTemplate, LLMChain

llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0, openai_api_key="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")

# Next, we define a prompt template for summarization:
summarization_template = "Summarize the following text to one sentence: {text}"
summarization_prompt = PromptTemplate(input_variables=["text"], template=summarization_template)
summarization_chain = LLMChain(llm=llm, prompt=summarization_prompt)

# To use the summarization chain, simply call the predictCopy method with the text to be summarized:
text = "LangChain provides many modules that can be used to build language model applications. Modules can be combined to create more complex applications, or be used individually for simple applications. The most basic building block of LangChain is calling an LLM on some input. Let’s walk through a simple example of how to do this. For this purpose, let’s pretend we are building a service that generates a company name based on what the company makes."
summarized_text = summarization_chain.predict(text=text)

