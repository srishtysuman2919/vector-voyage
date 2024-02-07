'''
    OpenAI's GPT-4 represents a significant advancement in the field of large language models. 
    Among its many improvements are enhanced creativity, the ability to process visual input, and an extended contextual understanding. 
    In the realm of conversational AI, both GPT-4 and ChatGPT use the Transformers architecture at their core and are fine-tuned to hold natural dialogue with a user. 
    While the free version of ChatGPT relies on GPT-3, the premium offering, ChatGPT Plus, gives access to the more advanced GPT-4 model.

    The benefits of employing ChatGPT and GPT-4 in chat format are numerous. 
    For instance, GPT-4's short-term memory capacity of 64,000 words greatly surpasses GPT-3.5's 8,000-word limit, 
    enabling it to maintain context more effectively in prolonged conversations. 
    Furthermore, GPT-4 is highly multilingual, accurately handling up to 26 languages, and boasts improved steering capabilities, 
    allowing users to tailor responses with a custom "personality."

    The new model is considerably safer to use, boasting a 40% increase in factual responses and an 82% reduction in disallowed content responses. 
    It can also interpret images as a foundation for interaction. 
    While this functionality has not yet been incorporated into ChatGPT, its potential to revolutionize context-aware chat applications is immense.
'''



# The following example demonstrates how to create a chatbot using the GPT-4 model from OpenAI. 
# After importing the necessary classes, we declare a set of messages. 
# It starts by setting the context for the model (SystemMessage) that it is an assistant, 
# followed by the user’s query (HumanMessage), 
# and finishes by defining a sample response from the AI model (AIMessage). 

from langchain.schema import (SystemMessage, HumanMessage, AIMessage)
from langchain.chat_models import ChatOpenAI

messages = [
    SystemMessage(content="You are a helpful assistant."),
    HumanMessage(content="What is the capital of France?"),
    AIMessage(content="The capital of France is Paris.")
]

# Next up, we test if the model can leverage these discussions as a reference to delve further into details about the city 
# without us explicitly mentioning the name (referring to Paris). 
# The code below adds a new message which requires the model to understand and find the “city you just mentioned” reference from previous conversations.
prompt = HumanMessage(
    content="I'd like to know more about the city you just mentioned."
)
# add to messages
messages.append(prompt)

llm = ChatOpenAI(model_name="gpt-4")

response = llm(messages)

#  The model successfully extracted the information from previous conversations and explained more details about Paris. 
# It shows that the chat models are capable of referring to the chat history and understanding the context.

'''
    To recap, the ChatOpenAI class is used to create a chat-based application that can handle user inputs and generate responses using the GPT-4 language model. 
    The conversation is initiated with a series of messages, including system, human, and AI messages. 
    The SystemMessage provides context for the conversation, while HumanMessage and AIMessage represent the user and the AI's messages, respectively.

    The LangChain’s Chat API offers several advantages:
    1. Context preservation: By maintaining a list of messages in the conversation, the API ensures that the context is preserved throughout the interaction. 
                             This allows the GPT-4 model to generate relevant and coherent responses based on the provided information.
    2. Memory: The class’s message history acts as a short-term memory for the chatbot, allowing it to refer back to previous messages and provide more accurate and 
               contextual responses.
    3. Modularity: The combination of MessageTemplate and ChatOpenAI classes offers a modular approach to designing conversation applications.
                   This makes it easier to develop, maintain, and extend the functionality of the chatbot.
    4. Improved performance: GPT-4, as an advanced language model, is more adept at understanding complex prompts and generating better responses than its predecessors. 
                             It can handle tasks that require deeper reasoning and context awareness, which leads to a more engaging and useful conversation experience.
    5. Flexibility: The Chat API can be adapted to different domains and tasks, making it a versatile solution for various chatbot applications. 
                    In this example, the chatbot specializes in French culture but could be easily modified to focus on other subjects or industries. 
                    Moreover, as newer and more powerful language models become available, the API can be updated to utilize those models, 
                    allowing for continuous improvements in chatbot capabilities.
'''
