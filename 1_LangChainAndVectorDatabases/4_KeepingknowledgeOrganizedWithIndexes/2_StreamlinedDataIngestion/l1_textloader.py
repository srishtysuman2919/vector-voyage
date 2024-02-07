from langchain.document_loaders import TextLoader

loader = TextLoader('my_file.txt')
documents = loader.load()