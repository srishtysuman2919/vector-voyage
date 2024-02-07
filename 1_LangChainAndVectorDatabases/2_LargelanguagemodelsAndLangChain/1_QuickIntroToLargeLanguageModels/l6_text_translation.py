'''
    It is one of the great attributes of Large Language models that enables them to perform multiple tasks just by changing the prompt. 
    We use the same llmCopy variable as defined before. 
    However, pass a different prompt that asks for translating the query from a source_languageCopy to the target_languageCopy.
'''

from langchain import PromptTemplate
from langchain import LLMChain
from langchain.llms import OpenAI


llm=OpenAI(model="text-davinci-003", temperature=0.9, openai_api_key="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")

translation_template = "Translate the following text from {source_language} to {target_language}: {text}"

translation_prompt = PromptTemplate(
    input_variables=["source_language", "target_language", "text"], 
    template=translation_template
    )

translation_chain = LLMChain(llm=llm, prompt=translation_prompt)

# To use the translation chain, call the predictCopy method with the source language, target language, and text to be translated:

source_language = "English"
target_language = "French"
text = "hey translate this language"
translated_text = translation_chain.predict(source_language=source_language, target_language=target_language, text=text)

