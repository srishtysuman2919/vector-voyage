'''
Agents - employing a language model as a reasoning mechanism and linking it with the key element - a tool.

two primary modes of operation to consider when employing an LLM: as content generators and as reasoning engines.
When used as a "content generator": the language model is asked to create content entirely from its internal knowledge base. This approach can lead to highly creative outputs but can also result in unverified information or 'hallucinations' due to the model's reliance on pre-trained knowledge.
When functioning as a "reasoning engine": the Agent acts more as an information manager rather than a creator, using tools as well, and planning the next actions to take


Tool: A tool's interface is typically a function that takes a string as an input and returns a string as an output. Eg; Google Search, a Database lookup, a Python REPL, or other chains. 
Language Learning Model: The language model that powers the agent.
Agent: They evaluate the situation and decide on the appropriate tools to use, if necessary.
    "Action Agents": These agents determine and execute a single action; suitable for smaller tasks,
    "Plan-and-Execute Agents": These agents first devise a plan comprising multiple actions and then execute each action sequentially;  help maintain long-term objectives and focus. 

high-level workflow of Action Agents would look something like this:
    The agent receives user input.
    It decides which tool to use (if any) and determines its input.
    The chosen tool is called with the provided input, and an observation (the output of the tool) is recorded.
    The history of the tool, tool input, and observation are relayed back to the agent, which then decides the next step.
    This process is repeated until the agent no longer needs to use a tool, at which point it directly responds to the user.

Tools are instrumental in connecting the language model with other sources of data or computation, including search engines, APIs, and other data repositories. 
Language models can only access the knowledge they've been trained on, which can quickly become obsolete. 
Therefore, tools are essential as they allow the agent to retrieve and incorporate current data into the prompt as context. 
Tools can also execute actions (like running code or modifying files) and observe the results, subsequently informing the language model's decision-making process.



'''


# Importing necessary modules
from langchain.agents import load_tools, initialize_agent
from langchain.agents import AgentType
from langchain.llms import OpenAI

# Loading the language model to control the agent
llm = OpenAI(model="text-davinci-003", temperature=0)

# Loading some tools to use. The llm-math tool uses an LLM, so we pass that in.
tools = load_tools(["google-search", "llm-math"], llm=llm)

# Initializing an agent with the tools, the language model, and the type of agent we want to use.
agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)

# Testing the agent
query = "What's the result of 1000 plus the number of goals scored in the soccer world cup in 2018?"
response = agent.run(query)
print(response)

# Importing necessary modules
from langchain.agents import initialize_agent, AgentType
from langchain.llms import OpenAI
from langchain.agents import Tool
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

prompt = PromptTemplate(
    input_variables=["query"],
    template="You're a renowned science fiction writer. {query}"
)

# Initialize the language model
llm = OpenAI(model="text-davinci-003", temperature=0)
llm_chain = LLMChain(llm=llm, prompt=prompt)

tools = [
    Tool(
        name='Science Fiction Writer',
        func=llm_chain.run,
        description='Use this tool for generating science fiction stories. Input should be a command about generating specific types of stories.'
    )
]

# Initializing an agent with the tools, the language model, and the type of agent we want to use.
agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)

# Testing the agent with the new prompt
response = agent.run("Compose an epic science fiction saga about interstellar explorers")
print(response)

from langchain.agents import Cus

