from langchain.embeddings import HuggingFaceEmbeddings
import chromadb
from chromadb.config import DEFAULT_TENANT, DEFAULT_DATABASE, Settings

# Initialize the HuggingFaceEmbeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Example text
texts = ["This is an example senyrrrrtytence.", "This is anotheytytr sentence.", "Hola Que passa Soy Omar"]

# Generate embeddings
embedding_vectors = embeddings.embed_documents(texts)

# Initialize PersistentClient with a persistent directory
client = chromadb.PersistentClient(
    path='C:\\Users\\omara\\Desktop\\Talan\\chromadb_data',
    settings=Settings(),
    tenant=DEFAULT_TENANT,
    database=DEFAULT_DATABASE
)

# Check if collection exists and create if not
collection_name = 'Testbiology_collection'
try:
    collection = client.get_collection(name=collection_name)
except KeyError:  # Assuming KeyError is raised if the collection does not exist
    collection = client.create_collection(name=collection_name)

# Prepare data for insertion
documents = [
    {"embedding": embedding, "metadata": {"text": text}}
    for embedding, text in zip(embedding_vectors, texts)
]

# Insert documents into the collection
collection.upsert(documents)

# Check the collection - Retrieve all documents
all_docs = collection.get_all()
for doc in all_docs:
    print(f"Embedding: {doc['embedding'][:10]}...")  # Print the first 10 dimensions for brevity
    print(f"Text: {doc['metadata']['text']}")