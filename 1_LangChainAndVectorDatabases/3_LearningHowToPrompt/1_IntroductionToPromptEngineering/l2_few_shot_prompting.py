'''
    Few Shot Prompting-> the LLM is asked to provide the emotion associated with a given color based on a few examples of color-emotion pairs
'''

from langchain.llms import OpenAI
from langchain import (PromptTemplate, FewShotPromptTemplate, LLMChain)

# Initialize LLM
llm = OpenAI(model_name="text-davinci-003", temperature=0, openai_api_key="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")

examples = [
    {"color": "red", "emotion": "passion"},
    {"color": "blue", "emotion": "serenity"},
    {"color": "green", "emotion": "tranquility"},
]

example_formatter_template = """ Color: {color} Emotion: {emotion}\n """
example_prompt = PromptTemplate(
    input_variables=["color", "emotion"],
    template=example_formatter_template,
)

few_shot_prompt = FewShotPromptTemplate(
    examples=examples,
    example_prompt=example_prompt,
    prefix="Here are some examples of colors and the emotions associated with them:\n\n",
    suffix="\n\nNow, given a new color, identify the emotion associated with it:\n\nColor: {input}\nEmotion:",
    input_variables=["input"],
    example_separator="\n",
)

formatted_prompt = few_shot_prompt.format(input="purple")
prompt=PromptTemplate(
    template=formatted_prompt, 
    input_variables=[]
    )

# Create the LLMChain for the prompt
chain = LLMChain(llm=llm, prompt=prompt)

# Run the LLMChain to get the AI-generated emotion associated with the input color
response = chain.run({})

print("Color: purple")
print("Emotion:", response)



