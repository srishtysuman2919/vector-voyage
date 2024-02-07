'''
    Emergent abilities, Scaling laws, and hallucinations:
        Another aspect of LLMs is their emergent abilities, which arise as a result of extensive pre-training on vast datasets. 
        These capabilities are not explicitly programmed but emerge as the model discerns patterns within the data. 
        LangChain models capitalize on these emergent abilities by working with various types of models, such as chat models and text embedding models. 
        This allows LLMs to perform diverse tasks, from answering questions to generating text and offering recommendations.

        Lastly, scaling laws describe the relationship between model size, training data, and performance. 
        Generally, as the model size and training data volume increase, so does the model's performance. 
        However, this improvement is subject to diminishing returns and may not follow a linear pattern. 
        It is essential to weigh the trade-off between model size, training data, performance, 
        and resources spent on training when selecting and fine-tuning LLMs for specific tasks.

        While Large Language Models boast remarkable capabilities but are not without shortcomings, 
        one notable limitation is the occurrence of hallucinations, in which these models produce text that appears plausible on the surface 
        but is actually factually incorrect or unrelated to the given input. 
        
        Additionally, LLMs may exhibit biases originating from their training data, resulting in outputs that can perpetuate stereotypes or generate undesired outcomes.
'''

# Creating a Question-Answering Prompt Template
# Let's create a simple question-answering prompt template using LangChain

from langchain import PromptTemplate
from langchain import HuggingFaceHub, LLMChain
from langchain.llms import OpenAI


template = """Question: {question}

Answer: """
prompt = PromptTemplate(
    template=template,
    input_variables=['question']
)

# user question
question = "What is the capital city of France?"

# Next, we will use the Hugging Face model google/flan-t5-largeCopy to answer the question. 
# The HuggingfaceHubCopy class will connect to Hugging Face’s inference API and load the specified model.


# initialize Hub LLM
hub_llm = HuggingFaceHub(
    repo_id='google/flan-t5-large',
    model_kwargs={'temperature':0},
    huggingfacehub_api_token="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"
)

# create prompt template > LLM chain
llm_chain = LLMChain(
    prompt=prompt,
    llm=hub_llm
)

# ask the user question about the capital of France
print(llm_chain.run(question))

# Asking Multiple Questions

# We can also modify the prompt template to include multiple questions.
# To ask multiple questions, we can either iterate through all questions one at a time or place all questions into a single prompt for more advanced LLMs. 
# Let's start with the first approach:

qa = [
    {'question': "What is the capital city of France?"},
    {'question': "What is the largest mammal on Earth?"},
    {'question': "Which gas is most abundant in Earth's atmosphere?"},
    {'question': "What color is a ripe banana?"}
]
res = llm_chain.generate(qa)
print( res )

# We can modify our prompt template to include multiple questions to implement a second approach. 
# The language model will understand that we have multiple questions and answer them sequentially. This method performs best on more capable models.

multi_template = """Answer the following questions one at a time.

Questions:
{questions}

Answers:
"""


long_prompt = PromptTemplate(template=multi_template, input_variables=["questions"])
llm = OpenAI(model="text-davinci-003", temperature=0.9, openai_api_key="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")

llm_chain = LLMChain(
    prompt=long_prompt,
    llm=llm
)

qs_str = (
    "What is the capital city of France?\n" +
    "What is the largest mammal on Earth?\n" +
    "Which gas is most abundant in Earth's atmosphere?\n" +
		"What color is a ripe banana?\n"
)
llm_chain.run(qs_str)

