'''
    LLM models operate on a straightforward principle: accept a text input sequence and generate an output text sequence. 
    Key factor: input text or prompt.

    Crafting suitable prompts is vital for anyone working with large language models: 
        poorly constructed prompts -> unsatisfactory outputs, 
        well-formulated prompts -> powerful results. 
'''

#   In this examples,
#   template-> formatted string with a {query} placeholder that will be substituted with a real question when applied. 
#   PromptTemplate object-> used to produce prompts with specific questions by providing input data. 
#       two arguments are required:
#           input_variables: A list of variable names in the template; in this case
#           template: The template string containing formatted text and placeholders.
#   Input data-> dictionary where the key corresponds to the variable name in the template. 
#   The resulting prompt can then be passed to a language model to generate answers.

from langchain.llms import OpenAI
from langchain import PromptTemplate, LLMChain

llm=OpenAI(model="text-davinci-003", temperature=0, openai_api_key="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")
template= """Answer the question based on the context below. If the
question cannot be answered using the information provided, answer
with "I don't know".
Context: Quantum computing is an emerging field that leverages quantum mechanics to solve complex problems faster than classical computers.
...
Question: {query}
Answer: """
prompt=PromptTemplate(template=template, input_variables=["query"])

# Create the LLMChain for the prompt
chain=LLMChain(prompt=prompt, llm=llm)

# Set the query you want to ask
input_data = {"query": "What is the main advantage of quantum computing over classical computing?"}

response=chain.run(input_data)

print("Question:", input_data["query"])
print("Answer:", response)


