'''
    LangChain provides a variety of tools for agents to interact with the outside world. 
    These tools can be used to create custom agents that perform various tasks, such as searching the web, answering questions, or running Python code. 
    In this section, we will discuss the different tool types available in LangChain and provide examples of creating and using them.

    In our example, two tools are being defined for use within a LangChain agent: a Google Search tool and a Language Model tool acting specifically as a text summarizer. 
    The Google Search tool, using the GoogleSearchAPIWrapper, will handle queries that involve finding recent event information. 
    The Language Model tool leverages the capabilities of a language model to summarize texts. 
    These tools are designed to be used interchangeably by the agent, depending on the nature of the user's query.
'''

from langchain.llms import OpenAI
from langchain.agents import Tool
from langchain.utilities import GoogleSearchAPIWrapper
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.agents import initialize_agent, AgentType

open_ai_token="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"
llm = OpenAI(model="text-davinci-003", temperature=0, openai_api_key=open_ai_token)

# We then instantiate a  LLMChain specifically for text summarization.
prompt = PromptTemplate(
    input_variables=["query"],
    template="Write a summary of the following text: {query}"
)

summarize_chain = LLMChain(llm=llm, prompt=prompt)

# remember to set the environment variables
# “GOOGLE_API_KEY” and “GOOGLE_CSE_ID” to be able to use
# Google Search via API.

GOOGLE_CSE_ID = "b3e121dac0f5b4f77"
GOOGLE_API_KEY = "AIzaSyCt-v9EmEu1zgsXMyzYk_xWYQksRcB05O8"
search = GoogleSearchAPIWrapper(google_api_key=GOOGLE_API_KEY, google_cse_id=GOOGLE_CSE_ID)

# Next, we create the tools that our agent will use
tools = [
    Tool(
        name="Search",
        func=search.run,
        description="useful for finding information about recent events"
    ),
    Tool(
       name='Summarizer',
       func=summarize_chain.run,
       description='useful for summarizing texts'
    )
]

# We are now ready to create our agent that leverages two tools
agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True  
)

# Let’s run the agent with a question about summarizing the latest news about the Mars rover
response = agent("What's the latest news about the Mars rover? Then please summarize the results.")
print(response['output'])

# Notice how the agents used at first the “Search” tool to look for recent information about the Mars rover and then used the “Summarizer” tool for writing a summary.
# LangChain provides an expansive toolkit that integrates various functions to improve the functionality of conversational agents. Here are some examples:
# SerpAPI: This tool is an interface for the SerpAPI search engine, allowing the agent to perform robust online searches to pull in relevant data for a conversation or task.
# PythonREPLTool: This unique tool enables the writing and execution of Python code within an agent. This opens up a wide range of possibilities for advanced computations and interactions within the conversation.
# If you wish to add more specialized capabilities to your LangChain conversational agent, the platform offers the flexibility to create custom tools. 
# By following the general tool creation guidelines provided in the LangChain documentation, you can develop tools tailored to the specific needs of your application.
