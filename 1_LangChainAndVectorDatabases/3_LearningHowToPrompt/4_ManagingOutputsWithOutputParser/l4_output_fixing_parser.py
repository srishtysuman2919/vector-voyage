'''
    OutputFixingParser: Add a layer on top of the model’s response to help fix the errors.
    
    The following approaches work with the PydanticOutputParser class since it is the only one with a validation method.
'''


# This method tries to fix the parsing error by looking at the model’s response and the previous parser. It uses a Large Language Model (LLM) to solve the issue.

from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import List

# Define your desired data structure.
class Suggestions(BaseModel):
    words: List[str] = Field(description="list of substitue words based on context")
    reasons: List[str] = Field(description="the reasoning of why this word fits the context")

parser = PydanticOutputParser(pydantic_object=Suggestions)

missformatted_output = '{"words": ["conduct", "manner"], "reasoning": ["refers to the way someone acts in a particular situation.", "refers to the way someone behaves in a particular situation."]}'

# parser.parse(missformatted_output)

# As you can see in the error message, the parser correctly identified an error in our sample response (missformatted_output) 
# since we used the word reasoning instead of the expected reasons key. The OutputFixingParser class could easily fix this error.

from langchain.llms import OpenAI
from langchain.output_parsers import OutputFixingParser

model = OpenAI(model_name='text-davinci-003', temperature=0.0, openai_api_key="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")

outputfixing_parser = OutputFixingParser.from_llm(parser=parser, llm=model)
outputfixing_parser.parse(missformatted_output)

# The from_llm() function takes the old parser and a language model as input parameters. 
# Then, It initializes a new parser for you that has the ability to fix output errors. 
# In this case, it successfully identified the misnamed key and changed it to what we defined.
# However, fixing the issues using this class is not always possible. Here is an example of using OutputFixingParser class to resolve an error with a missing key.

missformatted_output = '{"words": ["conduct", "manner"]}'

outputfixing_parser = OutputFixingParser.from_llm(parser=parser, llm=model)

outputfixing_parser.parse(missformatted_output)

# Looking at the output, it is evident that the model understood the key reasons missing from the response but didn’t have the context of the desired outcome. 
# It created a list with one entry, while we expect one reason per word. This is why we sometimes need to use the RetryOutputParser class.

