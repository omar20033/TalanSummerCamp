from langchain.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import re

# Define your URLs
urls = [
    "https://phys.org/news/2024-07-high-sensitivity-technique-mercury-soil.html"
]

# Load documents
docs = [WebBaseLoader(url).load() for url in urls]
docs_list = [item for sublist in docs for item in sublist]

# Function to preprocess content
def preprocess_content(text):
    # Remove ads, email forms, and other unrelated content
    text = re.sub(r"[\n\s]+", " ", text)  # Replace multiple spaces/newlines with a single space
    text = re.sub(r"(?i)(Email|Subscribe|Privacy|Donate|Topics Week\'s|top Latest|news Unread|news Subscribe|Science X|Feedback).*?[\n]", "", text)  # Remove common sections
    text = re.sub(r"(?i)more information.*", "", text)
    return text

# Preprocess documents
for doc in docs_list:
    doc.page_content = preprocess_content(doc.page_content)

# Initialize text splitter
text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=500, chunk_overlap=50
)

# Split documents into chunks
doc_splits = text_splitter.split_documents(docs_list)
print(doc_splits)