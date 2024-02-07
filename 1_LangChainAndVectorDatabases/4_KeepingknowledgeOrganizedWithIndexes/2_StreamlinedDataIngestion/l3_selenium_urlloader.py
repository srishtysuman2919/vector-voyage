'''
    The SeleniumURLLoader module offers a robust yet user-friendly approach for loading HTML documents from a list of URLs requiring JavaScript rendering.
'''

# Instantiate the SeleniumURLLoader class by providing a list of URLs to load, for example:

from langchain.document_loaders import SeleniumURLLoader

urls = [
    "https://www.youtube.com/watch?v=TFa539R09EQ&t=139s",
    "https://www.youtube.com/watch?v=6Zv6A_9urh4&t=112s"
]

loader = SeleniumURLLoader(urls=urls)
data = loader.load()

print(data[0])