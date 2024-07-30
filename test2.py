from langchain.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Load the document from the web
urls = ["https://phys.org/news/2024-07-images-shark-struck-boat.html"]
docs = [WebBaseLoader(url).load() for url in urls]
docs_list = [item for sublist in docs for item in sublist]
print(docs_list)







# Split the document into chunks
text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=250, chunk_overlap=0
)
doc_splits = text_splitter.split_documents(docs_list)

# Print the split documents for inspection
#for i, split in enumerate(doc_splits):
    #print(f"Chunk {i}:\n{split}\n")
    #print(len(doc_splits))