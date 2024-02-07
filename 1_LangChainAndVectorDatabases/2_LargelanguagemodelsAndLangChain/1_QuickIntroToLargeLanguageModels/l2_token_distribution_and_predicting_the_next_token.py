'''
    Large language models like GPT-3 and GPT-4 are pretrained on vast amounts of text data and learn to predict the next token in a sequence 
    based on the context provided by the previous tokens. GPT-family models use Causal Language modeling, 
    which predicts the next token while only having access to the tokens before it. This process enables LLMs to generate contextually relevant text.
'''

# The following code uses LangChain’s OpenAICopy class to load GPT-3’s Davinci variation using text-davinci-003Copy key to complete the sequence, which results in the answer. 

from langchain.llms import OpenAI

llm = OpenAI(model_name="text-davinci-003", temperature=0, openai_api_key="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")

text = "What would be a good company name for a company that makes colorful socks?"

print(llm(text))

# Tracking Token Usage
# You can use the LangChain library's callback mechanism to track token usage. This is currently implemented only for the OpenAI API:
from langchain.callbacks import get_openai_callback

llm = OpenAI(model_name="text-davinci-003", n=2, best_of=2, openai_api_key="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")

with get_openai_callback() as cb:
    result = llm("Tell me a joke")
    print(cb)

# The callback will track the tokens used, successful requests, and total cost.

