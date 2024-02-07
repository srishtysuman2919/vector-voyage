'''
    Primary goal of few-shot learning: learn a similarity function that maps the similarities between classes in the support and query sets. 

    Few-shot prompting: 
        Advantage: Improved output quality; since the model can learn the task better by observing the examples. 
        Disadvantage: Increased token usage may worsen the results if the examples are not well chosen or are misleading.

        Involves using the FewShotPromptTemplate class, which takes in a PromptTemplate and a list of a few shot examples. 
        The class formats the prompt template with a few shot examples, which helps the language model generate a better response. 
'''

from langchain import PromptTemplate, LLMChain, FewShotPromptTemplate
from langchain.chat_models import ChatOpenAI

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
example_template = """User: {query} AI: {answer}"""

# create a prompt example from above template
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

# After creating a template, we pass the example and user query, we get the results
llm=ChatOpenAI(model="gpt-3.5-turbo", temperature=0, openai_api_key="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")
chain=LLMChain(llm=llm, prompt=few_shot_prompt_template)
print(chain.run("What's the secret to happiness?"))

# This method allows for better control over example formatting and is suitable for diverse applications, 
# but it demands the manual creation of few-shot examples and can be less efficient with a large number of examples.



