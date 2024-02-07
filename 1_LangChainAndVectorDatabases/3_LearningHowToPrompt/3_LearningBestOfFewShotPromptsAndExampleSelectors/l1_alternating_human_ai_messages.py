'''
    Implementing Few-shot prompting and Example selection in LangChain can be achieved through various methods. 
        1. Alternating Human/AI messages
        2. Few-shot prompting
        3. Example selectors

    Alternating Human/AI messages: 
        Few-shot prompting utilizes alternating human and AI messages. 
        Advantages: Beneficial for chat-oriented applications.
        Disadvantage: Lacks flexibility for other application types and is limited to chat-based models.
'''
# we can use alternating human/AI messages to create a chat prompt that translates English into pirate language.

from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import (
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    AIMessagePromptTemplate,
    ChatPromptTemplate
)
from langchain import LLMChain

chat = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0, openai_api_key="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")

template="You are a helpful assistant that translates english to pirate."
system_message_prompt = SystemMessagePromptTemplate.from_template(template)
example_human = HumanMessagePromptTemplate.from_template("Hi")
example_ai = AIMessagePromptTemplate.from_template("Argh me mateys")
human_template="{text}"
human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, example_human, example_ai, human_message_prompt])
chain = LLMChain(llm=chat, prompt=chat_prompt)
chain.run("I love programming.")

