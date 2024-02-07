'''
    Chain Prompting-> chaining consecutive prompts, where the output of a previous prompt becomes the input of the successive prompt.  
    To use chain prompting with LangChain:
    1. Extract relevant information from the generated response.
    2. Use the extracted information to create a new prompt that builds upon the previous response.
    3. Repeat steps as needed until the desired output is achieved.
'''

from langchain.llms import OpenAI
from langchain import (
    PromptTemplate, LLMChain
    )

# Initialize LLM
llm = OpenAI(model_name="text-davinci-003", temperature=0, openai_api_key="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")

# Prompt 1
template_question = """What is the name of the famous scientist who developed the theory of general relativity?
Answer: """
prompt_question = PromptTemplate(
    template=template_question, 
    input_variables=[]
    )

# Prompt 2
template_fact = """Provide a brief description of {scientist}'s theory of general relativity.
Answer: """
prompt_fact = PromptTemplate(
    input_variables=["scientist"], 
    template=template_fact
    )

# Create the LLMChain for the first prompt
chain_question = LLMChain(llm=llm, prompt=prompt_question)

# Run the LLMChain for the first prompt with an empty dictionary
response_question = chain_question.run({})

# Extract the scientist's name from the response
scientist = response_question.strip()

# Create the LLMChain for the second prompt
chain_fact = LLMChain(llm=llm, prompt=prompt_fact)

# Input data for the second prompt
input_data = {"scientist": scientist}

# Run the LLMChain for the second prompt
response_fact = chain_fact.run(input_data)

print("Scientist:", scientist)
print("Fact:", response_fact)


'''
    Chain of Thought Prompting (CoT)-> encourage large language models to explain their reasoning process, leading to more accurate results. 
    By providing few-shot exemplars demonstrating the reasoning process, the LLM is guided to explain its reasoning when answering the prompt. 

    Advantages:
    1. Can help break down complex tasks by assisting the LLM in decomposing a complex task into simpler steps, making it easier to understand and solve the problem. 
    2. Guide the model through related prompts, helping generate more coherent and contextually relevant outputs. 

    Limitations:
    1. Yield performance gains only when used with models of approximately 100 billion parameters or larger; 
        smaller models tend to produce illogical chains of thought, which can lead to worse accuracy than standard prompting. 
    2. May not be equally effective for all tasks; most effective for tasks involving arithmetic, common sense, and symbolic reasoning. 
        For other types of tasks, the benefits of using CoT might be less pronounced or even counterproductive.
'''