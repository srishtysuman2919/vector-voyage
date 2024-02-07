from langchain.document_loaders import PyPDFLoader
loader = PyPDFLoader("/Users/srishtysuman/Downloads/cover_letter_wayfair_srishty_suman_data_scientist2.pdf")
pages = loader.load_and_split()

from langchain.text_splitter import CharacterTextSplitter

text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=20)
texts = text_splitter.split_documents(pages)

print(texts[0])

print (f"You have {len(texts)} documents")
print ("Preview:")
print (texts[0].page_content)