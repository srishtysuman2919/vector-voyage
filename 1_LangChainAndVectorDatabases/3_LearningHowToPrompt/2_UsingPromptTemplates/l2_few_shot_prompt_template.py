'''
    FewShotPromptTemplate-> dynamic prompt: incorporates examples of previous interactions, allowing the AI better to understand the context and style of the desired response.
    Advantages over static templates:
    1. Improved context understanding: By providing examples, the AI can grasp the context and style of responses more effectively, 
    2. Flexibility: Easily customizable
    3. Better results: Due to improved context understanding and flexibility, dynamic prompts tends to yields higher-quality outputs.

    Can create a FewShotPromptTemplate with an ExampleSelector to select a subset of examples that will be most informative for the language model.
'''

from langchain import LLMChain, FewShotPromptTemplate, PromptTemplate
from langchain.llms import OpenAI

llm=OpenAI(model="text-davinci-003", temperature=0, openai_api_key="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")

examples = [
    {"animal": "lion", "habitat": "savanna"},
    {"animal": "polar bear", "habitat": "Arctic ice"},
    {"animal": "elephant", "habitat": "African grasslands"}
]
example_template="""Animal:{animal}, Habitat:{habitat}"""

example_prompt=PromptTemplate(input_variables=["animal", "habitat"], template=example_template)

dynamic_prompt = FewShotPromptTemplate(
    examples=examples,
    example_prompt=example_prompt,
    prefix="Identify the habitat of the given animal",
    suffix="Animal: {input}\nHabitat:",
    input_variables=["input"],
    example_separator="\n\n",
)


chain=LLMChain(llm=llm, prompt=dynamic_prompt)

# Run the LLMChain with input_data
input_data = {"input": "tiger"}
response = chain.run(input_data)

print(response)

# Additionally, you can also save your PromptTemplate to a file in your local filesystem in JSON or YAML format:
dynamic_prompt.save("awesome_prompt.json")

# And load it back:
from langchain.prompts import load_prompt
loaded_prompt = load_prompt("awesome_prompt.json")

