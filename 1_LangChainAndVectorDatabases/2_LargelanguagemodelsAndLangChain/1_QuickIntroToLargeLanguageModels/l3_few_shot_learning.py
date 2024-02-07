'''
    Few-shot learning is a remarkable ability that allows LLMs to learn and generalize from limited examples. 
    Prompts serve as the input to these models and play a crucial role in achieving this feature. 
    With LangChain, examples can be hard-coded, but dynamically selecting them often proves more powerful, 
    enabling LLMs to adapt and tackle tasks with minimal training data swiftly.

    This approach involves using the FewShotPromptTemplateCopy class, which takes in a PromptTemplateCopy and a list of a few shot examples. 
    The class formats the prompt template with a few shot examples, which helps the language model generate a better response. 
    We can streamline this process by utilizing LangChain's FewShotPromptTemplate to structure the approach:
'''

from langchain import PromptTemplate
from langchain import FewShotPromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain import LLMChain

# create our examples
examples = [
    {
        "query": "What's the weather like?",
        "answer": "It's raining cats and dogs, better bring an umbrella!"
    }, {
        "query": "How old are you?",
        "answer": "Age is just a number, but I'm timeless."
    }
]


# create an example template
example_template = """
User: {query}
AI: {answer}
"""

# create a prompt example from above template (PronptTemplate also used in l2_chains.py)
example_prompt = PromptTemplate(
    input_variables=["query", "answer"],
    template=example_template
)

# now break our previous prompt into a prefix and suffix
# the prefix is our instructions
prefix = """The following are excerpts from conversations with an AI
assistant. The assistant is known for its humor and wit, providing
entertaining and amusing responses to users' questions. Here are some
examples:
"""
# and the suffix our user input and output indicator
suffix = """
User: {query}
AI: """

# now create the few-shot prompt template
few_shot_prompt_template = FewShotPromptTemplate(
    examples=examples,
    example_prompt=example_prompt,
    prefix=prefix,
    suffix=suffix,
    input_variables=["query"],
    example_separator="\n\n"
)

# After creating a template, we pass the example and user query, and we get the results
# load the model
chat = ChatOpenAI(model_name="gpt-4", temperature=0.0, openai_api_key="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")

chain = LLMChain(llm=chat, prompt=few_shot_prompt_template)
chain.run("What's the meaning of life?")

