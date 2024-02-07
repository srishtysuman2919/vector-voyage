'''
    CommaSeparatedListOutputParser-> manages comma-separated outputs. 
    It handles one specific case: anytime you want to receive a list of outputs from the model. 
'''

from langchain.output_parsers import CommaSeparatedListOutputParser
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate

# Loading OpenAI API
model = OpenAI(model_name='text-davinci-003', temperature=0.0, openai_api_key="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")

parser = CommaSeparatedListOutputParser()

# Prepare the Prompt
template = """
Offer a list of suggestions to substitute the word '{target_word}' based the presented the following text: {context}.
{format_instructions}
"""

prompt = PromptTemplate(
    template=template,
    input_variables=["target_word", "context"],
    partial_variables={"format_instructions": parser.get_format_instructions()}
)

model_input = prompt.format(
  target_word="behaviour",
  context="The behaviour of the students in the classroom was disruptive and made it difficult for the teacher to conduct the lesson."
)

# Send the Request
output = model(model_input)
parser.parse(output)

# The use of .format() instead of .format_prompt() to generate the model’s input. 
# The main difference compared to the previous subsection’s code is that we no longer need to call the .to_string() object since the prompt is already in string type.