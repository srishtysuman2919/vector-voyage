'''
    ConversationBufferMemory: TStores the entire conversation history as a single string. 
    Advantages: maintains a complete record of the conversation, as well as being straightforward to implement and use. 
    Disadvantages: 1. less efficient as the conversation grows longer 
                   2. may lead to excessive repetition if the conversation history is too long for the model's token limit.
                   3. the buffer gets truncated to fit within the model's token limit. 
                      => older interactions may be removed from the buffer to accommodate newer ones, 
                      => the conversation context might lose some information.

                    To avoid surpassing the token limit, we can monitor the token count in the buffer and manage the conversation accordingly. 
                    For example, we can choose to shorten the input texts or remove less relevant parts of the conversation to keep the token count within the model's limit.
'''


from langchain.memory import ConversationBufferMemory
from langchain.llms import OpenAI
from langchain.chains import ConversationChain

# TODO: Set your OPENAI API credentials in environemnt variables.
llm = OpenAI(model_name="text-davinci-003", temperature=0)

conversation = ConversationChain(
    llm=llm, 
    verbose=True, 
    memory=ConversationBufferMemory()
)
conversation.predict(input="Hello!")

from langchain import OpenAI, LLMChain, PromptTemplate
from langchain.memory import ConversationBufferMemory

template = """You are a customer support chatbot for a highly advanced customer support AI 
for an online store called "Galactic Emporium," which specializes in selling unique,
otherworldly items sourced from across the universe. You are equipped with an extensive
knowledge of the store's inventory and possess a deep understanding of interstellar cultures. 
As you interact with customers, you help them with their inquiries about these extraordinary
products, while also sharing fascinating stories and facts about the cosmos they come from.

{chat_history}
Customer: {customer_input}
Support Chatbot:"""

prompt = PromptTemplate(
    input_variables=["chat_history", "customer_input"], 
    template=template
)
chat_history=""

convo_buffer = ConversationChain(
    llm=llm,
    memory=ConversationBufferMemory()
)

print(conversation.prompt.template)

convo_buffer("I'm interested in buying items from your store")
convo_buffer("I want toys for my pet, do you have those?")
convo_buffer("I'm interested in price of a chew toys, please")

'''
    Token Count: The cost of utilizing the AI model in ConversationBufferMemory is directly influenced by the number of tokens used in a conversation, 
    more tokens => more expensive API requests.
'''
import tiktoken

def count_tokens(text: str) -> int:
    tokenizer = tiktoken.encoding_for_model("gpt-3.5-turbo")
    tokens = tokenizer.encode(text)
    return len(tokens)

conversation = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Who won the world series in 2020?"},
    {"role": "assistant", "content": "The Los Angeles Dodgers won the World Series in 2020."},
]

total_tokens = 0
for message in conversation:
    total_tokens += count_tokens(message["content"])

print(f"Total tokens in the conversation: {total_tokens}")


'''
    Strategies for managing tokens effectively: control the token count while minimizing associated costs and computational demands
    1. ConversationBufferWindowMemory: limits memory size by keeping a list of the most recent K interactions. 
                                       maintains a sliding window of these recent interactions, ensuring that the buffer does not grow too large. 
                                       stores a fixed number of recent messages in the conversation that makes it more efficient than ConversationBufferMemory.
                                       reduces the risk of exceeding the model's token limit. 
                                       Disadvantage: might lose context if essential information falls outside the fixed window of messages.

    2. ConversationSummaryBufferMemory: combines the ideas of keeping a buffer of recent interactions in memory and compiling old interactions into a summary. 
                                        uses token length rather than the number of interactions to determine when to flush interactions. 
                                        Advantages:
                                        1. Condensing conversation information
                                        2. Flexibility: can configure this type of memory to return the history as a list of messages or as a plain text summary.
                                        3. Direct summary prediction: directly obtain a summary prediction based on the list of messages and the previous summary.
                                        Disadvantages:
                                        1. Loss of information: summarizing the conversation might lead to a loss of information
                                        2. Increased complexity: requires more tweaking on what to summarize and what to maintain within the buffer window
'''
from langchain.memory import ConversationBufferWindowMemory
from langchain import OpenAI, LLMChain, PromptTemplate

template = """You are ArtVenture, a cutting-edge virtual tour guide for
 an art gallery that showcases masterpieces from alternate dimensions and
 timelines. Your advanced AI capabilities allow you to perceive and understand
 the intricacies of each artwork, as well as their origins and significance in
 their respective dimensions. As visitors embark on their journey with you
 through the gallery, you weave enthralling tales about the alternate histories
 and cultures that gave birth to these otherworldly creations.

{chat_history}
Visitor: {visitor_input}
Tour Guide:"""

prompt = PromptTemplate(
    input_variables=["chat_history", "visitor_input"], 
    template=template
)

chat_history=""

convo_buffer_win = ConversationChain(
    llm=llm,
    memory = ConversationBufferWindowMemory(k=3, return_messages=True)
)

convo_buffer_win("What is your name?")
convo_buffer_win("What can you do?")
convo_buffer_win("Do you mind give me a tour, I want to see your galery?")
convo_buffer_win("what is your working hours?")
convo_buffer_win("See you soon.")

from langchain.chains import ConversationChain
from langchain.memory import ConversationSummaryMemory

# Create a ConversationChain with ConversationSummaryMemory
conversation_with_summary = ConversationChain(
    llm=llm, 
    memory=ConversationSummaryMemory(llm=llm),
    verbose=True
)

# Example conversation
response = conversation_with_summary.predict(input="Hi, what's up?")
print(response)

from langchain.prompts import PromptTemplate

prompt = PromptTemplate(
    input_variables=["topic"],
    template="The following is a friendly conversation between a human and an AI. The AI is talkative and provides lots of specific details from its context. If the AI does not know the answer to a question, it truthfully says it does not know.\nCurrent conversation:\n{topic}",
)

from langchain.llms import OpenAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationSummaryBufferMemory

llm = OpenAI(temperature=0)
conversation_with_summary = ConversationChain(
    llm=llm,
    memory=ConversationSummaryBufferMemory(llm=OpenAI(), max_token_limit=40),
    verbose=True
)
conversation_with_summary.predict(input="Hi, what's up?")
conversation_with_summary.predict(input="Just working on writing some documentation!")
response = conversation_with_summary.predict(input="For LangChain! Have you heard of it?")
print(response)

'''
Summary: If the ConversationBufferMemory surpasses the token limit of the model:
            "model will not be able to handle the conversation with the exceeded token count"
            "adopt different strategies:"
            1. Remove oldest messages
            2. Limit conversation duration
            3. ConversationBufferWindowMemory Method:
            4. ConversationSummaryBufferMemory Approach: ConversationSummaryMemory + ConversationBufferWindowMemory.
'''