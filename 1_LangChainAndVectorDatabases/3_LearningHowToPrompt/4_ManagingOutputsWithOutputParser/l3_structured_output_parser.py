'''
    StructuredOutputParser: 
    1. It can process multiple outputs 
    2. It only supports texts and does not provide options for other data types, such as lists or integers. 
    3. It can be used when we want to receive one response from the model. For example, only one substitute word in the thesaurus application.
'''

from langchain.output_parsers import StructuredOutputParser, ResponseSchema

response_schemas = [
    ResponseSchema(name="words", description="A substitue word based on context"),
    ResponseSchema(name="reasons", description="The reasoning of why this word fits the context.")
]

parser = StructuredOutputParser.from_response_schemas(response_schemas)

# This class has no advantage since the PydanticOutputParser class provides validation and more flexibility for more complex tasks, 
# and the CommaSeparatedOutputParser option covers more straightforward applications.