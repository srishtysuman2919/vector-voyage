'''
    The LangChain library provides two methods for loading and processing PDF files: PyPDFLoader and PDFMinerLoader. 
    PyPDFLoader: load PDF files into an array of documents, where each document contains the page content and metadata with the page number. 
            advantages: simple, straightforward usage and easy access to page content and metadata, like page numbers, in a structured format. 
            disadvantages: limited text extraction capabilities compared to PDFMinerLoader.
'''


from langchain.document_loaders import PyPDFLoader

loader = PyPDFLoader("/Users/srishtysuman/Downloads/cover_letter_wayfair_srishty_suman_data_scientist2.pdf")
pages = loader.load_and_split()

print(pages[0])