'''
    ConversationChain: by default, it has a short-term memory that remembers all previous inputs/outputs and adds them to the context that is passed. 
'''


from langchain import OpenAI, ConversationChain

llm = OpenAI(model_name="text-davinci-003", temperature=0)
conversation = ConversationChain(llm=llm, verbose=True)

output = conversation.predict(input="Hi there!")

print(output)

# use the same conversation object to keep interacting with the model and ask various questions.
output = conversation.predict(input="In what scenarios extra memory should be used?")
output = conversation.predict(input="There are various types of memory in Langchain. When to use which type?")
output = conversation.predict(input="Do you remember what was our first message?")

print(output)

from langchain.memory import ConversationBufferMemory


# ConversationBufferMemory: used by default by ConversationChain, to provide a history of messages, save the previous conversations in form of variables. 
# The class accepts the return_messages argument which is helpful for dealing with chat models.

memory = ConversationBufferMemory(return_messages=True)
memory.save_context({"input": "hi there!"}, {"output": "Hi there! It's nice to meet you. How can I help you today?"})

print( memory.load_memory_variables({}) )

from langchain.chains import ConversationChain

# code in the previous section is the same as the following. It will automatically call the .save_context() object after each interaction.
conversation = ConversationChain(
    llm=llm, 
    verbose=True, 
    memory=ConversationBufferMemory()
)

from langchain import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder, SystemMessagePromptTemplate, HumanMessagePromptTemplate


# MessagesPlaceholder function: to create a placeholder for the conversation history in a chat model prompt
#                               particularly useful when working with ConversationChain and ConversationBufferMemory to maintain the context of a conversation. 
#                               takes a variable name as an argument, which is used to store the conversation history in the memory buffer. 

prompt = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template("The following is a friendly conversation between a human and an AI."),
    MessagesPlaceholder(variable_name="history"),
    HumanMessagePromptTemplate.from_template("{input}")
])

memory = ConversationBufferMemory(return_messages=True)
conversation = ConversationChain(memory=memory, prompt=prompt, llm=llm)


print( conversation.predict(input="Tell me a joke about elephants") )
print( conversation.predict(input="Who is the author of the Harry Potter series?") )
print( conversation.predict(input="What was the joke you told me earlier?") )

from langchain import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder, SystemMessagePromptTemplate, HumanMessagePromptTemplate

prompt = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template("The following is a friendly conversation between a human and an AI."),
    MessagesPlaceholder(variable_name="history"),
    HumanMessagePromptTemplate.from_template("{input}")
])

memory = ConversationBufferMemory(return_messages=True)
conversation = ConversationChain(memory=memory, prompt=prompt, llm=llm, verbose=True)

user_message = "Tell me about the history of the Internet."
response = conversation(user_message)
print(response)

# User sends another message
user_message = "Who are some important figures in its development?"
response = conversation(user_message)
print(response)  # Chatbot responds with names of important figures, recalling the previous topic

user_message = "What did Tim Berners-Lee contribute?"
response = conversation(user_message)
print(response)

