'''
The fundamental component of LangChain involves invoking an LLM with a specific input.
'''

# import llm rappper
from langchain.llms import OpenAI


# from dotenv import load_dotenv
# load_dotenv('/Users/srishtysuman/.env')
# print(load_dotenv('/Users/srishtysuman/.env'))

# The temperature parameter in OpenAI models manages the randomness of the output. 
# When set to 0, the output is mostly predetermined and suitable for tasks requiring stability and the most probable result. 
# At a setting of 1.0, the output can be inconsistent and interesting but isn't generally advised for most tasks. 
# For creative tasks, a temperature between 0.70 and 0.90 offers a balance of reliability and creativity. 
# The best setting should be determined by experimenting with different values for each specific use case. 

# The code initializes the GPT-3 model’s Davinci variant.
llm = OpenAI(model="davinci", temperature=0.9, openai_api_key="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")

# print(help(OpenAI))
print(llm)

# Now we can call it on some input!
text = "Suggest a personalized workout routine for someone looking to improve cardiovascular endurance and prefers outdoor activities."
print(llm(text))








