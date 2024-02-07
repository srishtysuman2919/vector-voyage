from langchain import ChatMessagePromptTemplate
prompt = "May the {subject} be with you"
chat_message_prompt = ChatMessagePromptTemplate.create_from_template(role="Jedi", template=prompt)
chat_message_prompt.format(subject="force")