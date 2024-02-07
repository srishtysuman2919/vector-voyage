from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

loader = PyPDFLoader("/Users/srishtysuman/Downloads/cover_letter_wayfair_srishty_suman_data_scientist2.pdf")
pages = loader.load_and_split()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=50,
    chunk_overlap=10,
    length_function=len,
)

docs = text_splitter.split_documents(pages)
for doc in docs:
    print(doc)