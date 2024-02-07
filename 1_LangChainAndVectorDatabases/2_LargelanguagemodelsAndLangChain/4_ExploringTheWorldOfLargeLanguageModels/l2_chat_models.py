'''
    Chat Models are the most popular models in LangChain, such as ChatGPT that can incorporate GPT-3 or GPT-4 at its core. 
    They have gained significant attention due to their ability to learn from human feedback and their user-friendly chat interface.

    Chat Models, such as ChatGPT, take a list of messages as input and return an AIMessage. 
    They typically use LLMs as their underlying technology, but their APIs are more structured. 
    Chat Models are designed to remember previous exchanges with the user in a session and use that context to generate more relevant responses. 
    They also benefit from reinforcement learning from human feedback, which helps improve their responses. 
    However, they may still have limitations in reasoning and require careful handling to avoid generating inappropriate content.

    In LangChain, three main types of messages are used when interacting with chat models: SystemMessage, HumanMessage, and AIMessage.
        SystemMessage: These messages provide initial instructions, context, or data for the AI model. 
                       They set the objectives the AI should follow and can help in controlling the AI's behavior.
                       System messages are not user inputs but rather guidelines for the AI to operate within.

                       SystemMessage represents the messages generated by the system that wants to use the model, 
                       which could include instructions, notifications, or error messages. 
                       These messages are not generated by the human user or the AI chatbot but are instead produced by the underlying system 
                       to provide context, guidance, or status updates.

        HumanMessage: These messages come from the user and represent their input to the AI model. 
                      The AI model is expected to respond to these messages. 
                      In LangChain, you can customize the human prefix (e.g., "User") in the conversation summary to change how the human input is represented.

        AIMessage: These messages are sent from the AI's perspective as it interacts with the human user. 
                   They represent the AI's responses to human input. 
                   Like HumanMessage, we can also customize the AI prefix (e.g., "AI Assistant" or "AI") in the conversation summary to change how the AI's responses are represented.
''' 

# In this section, we are trying to use the LangChain library to create a chatbot that can translate an English sentence into French. 
# This particular use case goes beyond what we covered in the previous lesson. 
# We'll be employing multiple message types to differentiate between users' queries and system instructions instead of relying on a single prompt. 
# This approach will enhance the model's comprehension of the given requirements.

# First, we create a list of messages, starting with a SystemMessage that sets the context for the chatbot, 
# informing it that its role is to be a helpful assistant translating English to French. 
# We then follow it with a HumanMessage containing the user’s query, like an English sentence to be translated.

from langchain.chat_models import ChatOpenAI
from langchain.schema import (SystemMessage, HumanMessage)

chat = ChatOpenAI(model_name="gpt-4", temperature=0, openai_api_key="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")

messages = [
	SystemMessage(content="You are a helpful assistant that translates English to French."),
	HumanMessage(content="Translate the following sentence: I love programming.")
]

chat(messages)

# we pass the list of messages to the chatbot using the chat() function. 
# The chatbot processes the input messages, considers the context provided by the system message, and then translates the given English sentence into French. 

# Using the generate method, you can also generate completions for multiple sets of messages. 
# Each batch of messages can have its own SystemMessage and will perform independently. 
# The following code shows the first set of messages translate the sentences from English to French, while the second ones do the opposite.

batch_messages = [
  [
    SystemMessage(content="You are a helpful assistant that translates English to French."),
    HumanMessage(content="Translate the following sentence: I love programming.")
  ],
  [
    SystemMessage(content="You are a helpful assistant that translates French to English."),
    HumanMessage(content="Translate the following sentence: J'aime la programmation.")
  ],
]
print( chat.generate(batch_messages) )

# LLMs and Chat Models each have their advantages and disadvantages. LLMs are powerful and flexible, capable of generating text for a wide range of tasks. 
# However, their API is less structured compared to Chat Models.

# On the other hand, Chat Models offer a more structured API and are better suited for conversational tasks. 
# Also, they can remember previous exchanges with the user, making them more suitable for engaging in meaningful conversations. 
# Additionally, they benefit from reinforcement learning from human feedback, which helps improve their responses. 
# They still have some limitations in reasoning and may require careful handling to avoid hallucinations and generating inappropriate content.



