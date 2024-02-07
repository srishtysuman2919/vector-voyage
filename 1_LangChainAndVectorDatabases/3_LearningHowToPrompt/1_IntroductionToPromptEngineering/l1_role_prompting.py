'''
    Prompt engineering-> involves developing and optimizing prompts to use language models.
    1. Role Prompting-> asking the LLM to assume a specific role or identity before performing a given task, such as acting as a copywriter. 
        1. Specify the role in your prompt, e.g., "As a copywriter, create some attention-grabbing taglines for AWS services."
        2. Use the prompt to generate an output from an LLM.
        3. Analyze the generated response and, if necessary, refine the prompt for better results.
'''

# In this example, the LLM is asked to act as a futuristic robot band conductor and suggest a song title related to the given theme and year

from langchain.llms import OpenAI
from langchain import PromptTemplate, LLMChain

template = """
As a futuristic robot band conductor, I need you to help me come up with a song title.
What's a cool song title for a song about {theme} in the year {year}?
"""
prompt = PromptTemplate(
    input_variables=["theme", "year"],
    template=template,
)

# Create the LLMChain for the prompt
llm = OpenAI(model_name="text-davinci-003", temperature=0, openai_api_key="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")

# Input data for the prompt
input_data = {"theme": "interstellar travel", "year": "3030"}

# Create LLMChain
chain = LLMChain(llm=llm, prompt=prompt)

# Run the LLMChain to get the AI-generated song title
response = chain.run(input_data)

print("Theme: interstellar travel")
print("Year: 3030")
print("AI-generated song title:", response)

# Advantages:
# 1. Clear instructions: This helps the LLM understand that the desired output should be a song title related to a futuristic scenario.
# 2. Specificity: This provides enough context for the LLM to generate a relevant and creative output.
# 3. Open-ended creativity: It doesn't limit the LLM to a particular format or style for the song title. 
# 4. Focus on the task: The prompt is focused solely on generating a song title, making it easier for the LLM to provide a suitable output.
# These elements help the LLM understand the user's intention and generate a suitable response.