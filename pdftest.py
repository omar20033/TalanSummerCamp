from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import re
from langchain_community.vectorstores import Chroma

from langchain_nomic.embeddings import NomicEmbeddings
pdf_path = r"C:\Users\omara\Desktop\Talan\pdftestttt.pdf"

# Load documents
loader = PyPDFLoader(pdf_path)
docs = loader.load()

def preprocess_content(text):
    # Remove newline characters
    text = text.replace('\n', ' ')
    # Optionally, you can add more preprocessing steps here
    return text

# Apply preprocessing to the loaded documents
for doc in docs:
    doc.page_content = preprocess_content(doc.page_content)


    
# Initialize text splitter
text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=500, chunk_overlap=50
)

# Split documents into chunks
doc_splits = text_splitter.split_documents(docs)
local_directory = r"C:\Users\omara\Desktop\Talan\chromaDb"
# Add to vectorDB
vectorstore = Chroma.from_documents(
    documents=doc_splits,
    collection_name="rag-chroma",
    embedding=NomicEmbeddings(model="nomic-embed-text-v1.5", inference_mode="local"),
    persist_directory=local_directory
)