'''
    Large Language Models have made significant advancements in the field of Natural Language Processing (NLP), enabling AI systems to understand and generate human-like text. 
    ChatGPT is a popular language model based on Transformers architecture, enabling it to understand long texts and figure out how words or ideas are connected. 
    It's great at making predictions about language and relationships between words.

    LLMs and Chat Models are two types of models in LangChain, serving different purposes in natural language processing tasks.

    LLMs, such as GPT-3, Bloom, PaLM, and Aurora genAI, take a text string as input and return a text string as output. 
    They are trained on language modeling tasks and can generate human-like text, perform complex reasoning, and even write code. 
    LLMs are powerful and flexible, capable of generating text for a wide range of tasks. 
    However, they can sometimes produce incorrect or nonsensical answers, and their API is less structured compared to Chat Models.

    Pre-training these models involves presenting large-scale corpora to them and allowing the network to predict the next word, 
    which results in learning the relationships between words. This learning process enables LLMs to generate high-quality text, 
    which can be applied to an array of applications, such as automatic form-filling and predictive text on smartphones.

    Most of these models are trained on general purpose training dataset, while others are trained on a mix of general and domain-specific data, 
    such as Intel Aurora genAI, which is trained on general text, scientific texts, scientific data, and codes related to the domain. 
    The goal of domain specific LLMs is to increase the performance on a particularly domain, while still being able to solve the majority of tasks that general LLM can manage.

    LLMs have the potential to infiltrate various aspects of human life, including the arts, sciences, and law. 
    With continued development, LLMs will become increasingly integrated into our educational, personal, and professional lives, making them an essential technology to master.
'''

# Import the OpenAI wrapper from the langchain.llms module and Initialize it with the desired model name and any additional arguments. 
# For example, set a high temperature for more random outputs. Then, create a PromptTemplate to format the input for the model. 
# Lastly, define an LLMChain to combine the model and prompt. Run the chain with the desired input using .run()

from langchain.llms import OpenAI
from langchain import PromptTemplate, LLMChain

llm=OpenAI(model="text-davinci-003", temperature=0, openai_api_key="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")
prompt=PromptTemplate(input_variables=["product"], template="What is a good name for a company that makes {product}?")
chain=LLMChain(llm=llm, prompt=prompt)
print(chain.run("wireless headphones"))

# Here, the input for the chain is the string "wireless headphones". The chain processes the input and generates a result based on the product name.