'''
    Primary goal of few-shot learning: learn a similarity function that maps the similarities between classes in the support and query sets. 
    Example selector can be designed to:
        1. choose a set of relevant examples that are representative of the desired output.
        2. select a subset of examples that will be most informative for the language model. 
    LengthBasedExampleSelector-> useful when concerned about the length of the context window
                                 selects fewer examples for longer queries and more examples for shorter queries.
'''

from langchain.prompts import PromptTemplate, FewShotPromptTemplate
from langchain.prompts.example_selector import LengthBasedExampleSelector
from langchain.vectorstores import DeepLake
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.prompts.example_selector import SemanticSimilarityExampleSelector

# Define your examples and the example_prompt
examples = [
    {"word": "happy", "antonym": "sad"},
    {"word": "tall", "antonym": "short"},
    {"word": "energetic", "antonym": "lethargic"},
    {"word": "sunny", "antonym": "gloomy"},
    {"word": "windy", "antonym": "calm"},
]
example_template=""" Word: {word}, Antonym: {antonym} """

example_prompt=PromptTemplate(input_variables=["word", "antonym"], template=example_template)

# Create an instance of LengthBasedExampleSelector
example_selector=LengthBasedExampleSelector(examples=examples, example_prompt=example_prompt, max_length=25)

# Create a FewShotPromptTemplate
dynamic_prompt=FewShotPromptTemplate(
    example_selector=example_selector, 
    example_prompt=example_prompt, 
    prefix="Give the antonym of every input",
    suffix="Word: {input}\nAntonym",
    input_variables=["input"],
    example_separator="\n\n",
    )

# Generate a prompt using the format method
print(dynamic_prompt.format(input="big"))

# Advantages: Offers customization through various selectors and effective for managing a large number of examples
# Disadvantages: Involves manual creation and selection of examples


'''
    SemanticSimilarityExampleSelector ->  Selects examples based on their semantic resemblance to the input. 
'''

# Create a PromptTemplate
example_prompt = PromptTemplate(
    input_variables=["input", "output"],
    template="Input: {input}\nOutput: {output}",
)

# Define some examples
examples = [
    {"input": "0°C", "output": "32°F"},
    {"input": "10°C", "output": "50°F"},
    {"input": "20°C", "output": "68°F"},
    {"input": "30°C", "output": "86°F"},
    {"input": "40°C", "output": "104°F"},
]


# create Deep Lake dataset
my_activeloop_org_id = "srishtysuman2919"
my_activeloop_dataset_name = "langchain_course_from_zero_to_hero"
dataset_path = "https:/app.activeloop.ai/srishtysuman2919/langchain_course_from_zero_to_hero"
deep_lake_token="eyJhbGciOiJIUzUxMiIsImlhdCI6MTcwMzk1NDQzMCwiZXhwIjoxNzM1NTc2ODE2fQ.eyJpZCI6InNyaXNodHlzdW1hbjI5MTkifQ.j_B5pt-7UL48fhp5cXStId-DNydAiuqDV3KlUC9cgE_ivDRyPaCmNnrWyTKp7Qkjx_rFL3M_T1yUR9HhG5vxmA"
db = DeepLake(dataset_path=dataset_path, token=deep_lake_token)

# Embedding function
embeddings = OpenAIEmbeddings(model="text-embedding-ada-002", openai_api_key="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")

# Instantiate SemanticSimilarityExampleSelector using the examples
example_selector = SemanticSimilarityExampleSelector.from_examples(
    examples, embeddings, db, k=1
)

# Create a FewShotPromptTemplate using the example_selector
similar_prompt = FewShotPromptTemplate(
    example_selector=example_selector,
    example_prompt=example_prompt,
    prefix="Convert the temperature from Celsius to Fahrenheit",
    suffix="Input: {temperature}\nOutput:", 
    input_variables=["temperature"],
)

# Test the similar_prompt with different inputs
print(similar_prompt.format(temperature="10°C"))   # Test with an input
print(similar_prompt.format(temperature="30°C"))  # Test with another input

# Add a new example to the SemanticSimilarityExampleSelector
similar_prompt.example_selector.add_example({"input": "50°C", "output": "122°F"})
print(similar_prompt.format(temperature="40°C")) # Test with a new input after adding the example


# SemanticSimilarityExampleSelector-> 1. uses the Deep Lake vector store and OpenAIEmbeddings to measure semantic similarity. 
#                                     2. stores the samples on the database in the cloud, and retrieves similar samples.
