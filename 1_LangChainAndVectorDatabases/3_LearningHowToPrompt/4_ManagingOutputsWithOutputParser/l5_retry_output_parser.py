'''
    RetryWithErrorOutputParser: 1. fix the same missformatted_output
                                2. receives the old parser and a model to declare the new parser object
                                3. fix the issue where the OuputFixingParser was not able to. 
                                4. best practice to incorporate these techniques in production is to catch the parsing error using a try: ... except: ... method. 
                                5. capture the errors in the except section and attempt to fix them using the mentioned classes. 
                                6. It will limit the number of API calls and avoid unnecessary costs that are associated with it
'''
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import List

# Define data structure.
class Suggestions(BaseModel):
    words: List[str] = Field(description="list of substitue words based on context")
    reasons: List[str] = Field(description="the reasoning of why this word fits the context")

parser = PydanticOutputParser(pydantic_object=Suggestions)

# Define prompt
template = """
Offer a list of suggestions to substitue the specified target_word based the presented context and the reasoning for each word.
{format_instructions}
target_word={target_word}
context={context}
"""

prompt = PromptTemplate(
    template=template,
    input_variables=["target_word", "context"],
    partial_variables={"format_instructions": parser.get_format_instructions()}
)

model_input = prompt.format_prompt(target_word="behaviour", context="The behaviour of the students in the classroom was disruptive and made it difficult for the teacher to conduct the lesson.")

# Define Model
model = OpenAI(model_name='text-davinci-003', temperature=0.0, openai_api_key="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")

# parse_with_prompt-> fixes the parsing issue while requiring the output and the prompt.

from langchain.output_parsers import RetryWithErrorOutputParser

missformatted_output = '{"words": ["conduct", "manner"]}'

retry_parser = RetryWithErrorOutputParser.from_llm(parser=parser, llm=model)

retry_parser.parse_with_prompt(missformatted_output, model_input)

