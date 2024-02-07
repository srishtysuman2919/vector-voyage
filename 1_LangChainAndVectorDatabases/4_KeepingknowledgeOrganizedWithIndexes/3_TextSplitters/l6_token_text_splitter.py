'''
    Main advantage of using TokenTextSplitter over other text splitters, like CharacterTextSplitter: 
        It respects the token boundaries, ensuring that the chunks do not split tokens in the middle. 
        This can be particularly helpful in maintaining the semantic integrity of the text when working with language models and embeddings.

    Breaks down raw text strings into smaller pieces by initially converting the text into BPE(Byte Pair Encoding) tokens then divides these tokens into chunks. 
'''

from langchain.text_splitter import TokenTextSplitter

# Load a long document
with open('my_file.txt', encoding= 'unicode_escape') as f:
    sample_text = f.read()

# Initialize the TokenTextSplitter with desired chunk size and overlap
text_splitter = TokenTextSplitter(chunk_size=100, chunk_overlap=50)

# Split into smaller chunks
texts = text_splitter.split_text(sample_text)
print(texts[0])