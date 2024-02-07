'''
    LangChain is designed to assist developers in building end-to-end applications using language models. 
    It offers an array of tools, components, and interfaces that simplify the process of creating applications powered by large language models and chat models. 
    LangChain streamlines managing interactions with LLMs, chaining together multiple components, and integrating additional resources, such as APIs and databases. 
    Having gained a foundational understanding of the library in previous lesson, let's now explore various examples of utilizing prompts to accomplish multiple tasks.

    A key feature of LangChain is its support for prompts, which encompasses prompt management, prompt optimization, and a generic interface for all LLMs. 
    The framework also provides common utilities for working with LLMs.

    ChatPromptTemplate is used to create a structured conversation with the AI model, making it easier to manage the flow and content of the conversation. 
    In LangChain, message prompt templates are used to construct and work with prompts, allowing us to exploit the underlying chat model's potential fully.

    System and Human prompts differ in their roles and purposes when interacting with chat models. 
    SystemMessagePromptTemplate provides initial instructions, context, or data for the AI model, 
    while HumanMessagePromptTemplate are messages from the user that the AI model responds to.
'''
from langchain.chat_models import ChatOpenAI
from langchain import (SystemMessagePromptTemplate, ChatPromptTemplate, HumanMessagePromptTemplate)


chat = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0, openai_api_key="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")

template = "You are an assistant that helps users find information about movies."
system_message_prompt = SystemMessagePromptTemplate.from_template(template)
human_template = "Find information about the movie {movie_title}."
human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])

response = chat(chat_prompt.format_prompt(movie_title="Inception").to_messages())

print(response.content)

# Using the to_messages object in LangChain allows you to convert the formatted value of a chat prompt template into a list of message objects. 
# This is useful when working with chat models, as it provides a structured way to manage the conversation 
# and ensures that the chat model can understand the context and roles of the messages.



