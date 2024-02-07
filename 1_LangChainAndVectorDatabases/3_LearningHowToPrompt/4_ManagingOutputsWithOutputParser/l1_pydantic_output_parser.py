'''
    Output Parsers->  The parsers are powerful tools to dynamically extract the information from the prompt and validate it to some extent. 
                      Creates a data structure to define the expectations from the output precisely. 
                      Define a data schema to generate correctly formatted responses.

    PydanticOutputParser parser-> Most powerful and flexible wrapper. 
                    -> Instructs the model to generate its output in a JSON format and then extract the information from the response. 
                    -> Can treat the parser’s output as a list, meaning it will be possible to index through the results without worrying about formatting.
                    -> uses Pydantic library, which helps define and validate data structures in Python. 
                    -> enables us to characterize the expected output with a name, type, and description.
'''



from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field, validator
from typing import List


'''
    There are two essential parts to this class:
    Expected Outputs: Each output is defined by declaring a variable with desired type, like a list of strings (: List[str]), single string (: str).
        Also, It is required to write a simple explanation using the Field function’s description attribute to help the model during inference. 
    Validators: It is possible to declare functions to validate the formatting. We ensure that the first character is not a number in the sample code. 
        The function’s name is unimportant, but the @validator decorator must receive the same name as the variable you want to approve. 
        It is worth noting that the field variable inside the validator function will be a list if you specify it as one.
'''
# Define your desired data structure.
class Suggestions(BaseModel):
    words: List[str] = Field(description="list of substitue words based on context")

    # Throw error in case of receiving a numbered-list from API
    @validator('words')
    def not_start_with_number(cls, field):
        for item in field:
            if item[0].isnumeric():
                raise ValueError("The word can not start with numbers!")
        return field

parser = PydanticOutputParser(pydantic_object=Suggestions)

# We will pass the created class to the PydanticOutputParser wrapper to make it a LangChain parser object. The next step is to prepare the prompt.
from langchain.prompts import PromptTemplate

template = """
Offer a list of suggestions to substitue the specified target_word based the presented context.
{format_instructions}
target_word={target_word}
context={context}
"""

prompt = PromptTemplate(
    template=template,
    input_variables=["target_word", "context"],
    partial_variables={"format_instructions": parser.get_format_instructions()}
)

model_input = prompt.format_prompt(
			target_word="behaviour",
			context="The behaviour of the students in the classroom was disruptive and made it difficult for the teacher to conduct the lesson."
)

# template-> string that outlines our expectations for the model, including the expected formatting from the parser and the inputs. 
# PromptTemplate-> receives the template string with the details of each placeholder’s type. 
# input_variables-> value is initialized later on using the .format_prompt() function
# partial_variables-> initialized instantly. 

# The prompt can send the query to models like GPT using LangChain’s OpenAI wrapper.
from langchain.llms import OpenAI

# Before executing the following code, make sure to have
# your OpenAI key saved in the “OPENAI_API_KEY” environment variable.
model = OpenAI(model_name='text-davinci-003', temperature=0.0, openai_api_key="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")

output = model(model_input.to_string())

print(parser.parse(output))

# The parser object’s parse() function will convert the model’s string response to the format we specified. 

# Here is a sample code for Pydantic class to process multiple outputs. It requests the model to suggest a list of words and present the reasoning behind each proposition.
model = OpenAI(model_name='text-davinci-003', temperature=0.0, openai_api_key="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")

class SuggestionsMultiple(BaseModel):
    words: List[str] = Field(description="list of substitue words based on context")
    reasons: List[str] = Field(description="the reasoning of why this word fits the context")
    
    @validator('words')
    def not_start_with_number(cls, field):
      for item in field:
        if item[0].isnumeric():
          raise ValueError("The word can not start with numbers!")
      return field
    
    @validator('reasons')
    def end_with_dot(cls, field):
      for idx, item in enumerate( field ):
        if item[-1] != ".":
          field[idx] += "."
      return field

parser_multiple = PydanticOutputParser(pydantic_object=SuggestionsMultiple)
   
template_multiple = """
Offer a list of suggestions to substitute the specified target_word based on the presented context and the reasoning for each word.
{format_instructions}
target_word={target_word}
context={context}
"""

prompt_multiple = PromptTemplate(
    template=template_multiple,
    input_variables=["target_word", "context"],
    partial_variables={"format_instructions": parser_multiple.get_format_instructions()}
)

model_input = prompt_multiple.format_prompt(
			target_word="behaviour",
			context="The behaviour of the students in the classroom was disruptive and made it difficult for the teacher to conduct the lesson."
)


output = model(model_input.to_string())

print(parser_multiple.parse(output))
    
