'''
  A few notable examples of Tools in langchain:
  Google Search: uses the Google Search API to fetch relevant information from the web.
  Requests: employs the Python library "requests" to interact with web services, access APIs, or obtain data from different online sources.
  Python REPL: allows to execute Python code on-the-fly to perform calculations, manipulate data, or test algorithms.
  Wikipedia: uses the Wikipedia API to search and retrieve relevant articles, summaries, or specific information from the vast repository of knowledge on the Wikipedia platform.
  Wolfram Alpha: users can tap into the computational knowledge engine of Wolfram Alpha to answer complex questions, perform advanced calculations, or generate visual representations of data.

  Agent: A bot that acts using natural language instructions and can use tools to answer its queries. 
         Based on user input, it is also used to determine which actions to take and in what order. 
         An action can either be using a tool (such as a search engine or a calculator) and processing its output or returning a response to the user. 
         To create an agent in LangChain, we can use the initialize_agent function along with the load_tools function to prepare the tools the agent can use.   
'''


from langchain.agents import load_tools
from langchain.agents import initialize_agent
from langchain.agents import AgentType
from langchain import OpenAI
import os

llm = OpenAI(model_name="gpt-3.5-turbo-instruct", temperature=0, openai_api_key="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")
tools = load_tools(['serpapi', 'requests_all'], llm=llm, serpapi_api_key="fbb180a00edd8deb8568ca6b2b5dc5915bc93d80b66018ac6eee603d271c3f1f")
agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)

agent.run("tell me what is midjourney?")


# As a standalone utility:
# Using GoogleSearchAPIWrapper to receive k top search results given a query. 
from langchain. utilities import GoogleSearchAPIWrapper
search = GoogleSearchAPIWrapper(google_api_key="AIzaSyCt-v9EmEu1zgsXMyzYk_xWYQksRcB05O8", google_cse_id="b3e121dac0f5b4f77")
search.results("What is the capital of Spain?", 3)


# Initialize an agent and load the google-search tool for it to use. The agent will load the search results and provide them to the llm to answer our question. 
# The ZERO_SHOT_REACT_DESCRIPTION type gives the freedom to choose any of the defined tools to provide context for the model based on their description.
from langchain.llms import OpenAI
from langchain.agents import initialize_agent, load_tools, AgentType

llm = OpenAI(model_name="gpt-3.5-turbo-instruct", temperature=0, openai_api_key="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")
tools = load_tools(["google-search"])
agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True  
)
print( agent("What is the national drink in Spain?") )


# This wrapper accepts a URL as an input and efficiently retrieves data from the specified URL, allowing LLMs to obtain and process web-based content effortlessly.
from langchain.agents import AgentType
tools = load_tools(["requests_all"], llm=llm)
agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
  )

response = agent.run("Get the list of users at https://644696c1ee791e1e2903b0bb.mockapi.io/user and tell me the total number of users")

tools = load_tools(["python_repl"], llm=llm)
agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
  )

print( agent.run("Create a list of random strings containing 4 letters, list should contain 30 examples, and sort the list alphabetically") )

agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True  
)

tools = load_tools(["wikipedia"])

print( agent.run("What is Nostradamus know for") )

import os

from langchain. utilities.wolfram_alpha import WolframAlphaAPIWrapper

wolfram = WolframAlphaAPIWrapper(wolfram_alpha_appid="VVLQU8-46THE24Y97")
result = wolfram.run("What is 2x+5 = -3x + 7?")
print(result)  # Output: 'x = 2/5'

tools = load_tools(["wolfram-alpha"])

agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True  
)

print( agent.run("How many days until the next Solar eclipse") )

tools = load_tools(["wolfram-alpha", "wikipedia"], llm=llm)

agent = initialize_agent(
		tools,
		llm,
		agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
		verbose=True
	)

agent.run("Who is Olivia Wilde's boyfriend? What is his current age raised to the 0.23 power?")


