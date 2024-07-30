from langchain.embeddings import HuggingFaceEmbeddings
import chromadbbbb

# Initialize the HuggingFaceEmbeddings (note: this might be different from HuggingFaceBgeEmbeddings)
embeddings = HuggingFaceEmbeddings()

# Example text
texts = ["This is an example sentence.", "This is another sentence.","Hola Que passa Soy Omar "]

# Generate embeddings
embedding_vectors = embeddings.embed_documents(texts)  # Method might be `embed_documents` or similar

# Print the generated embeddings
for i, vector in enumerate(embedding_vectors):
    print(f"Text: {texts[i]}")
    print(f"Embedding: {vector[:10]}...")  # Print the first 10 dimensions for brevity
    print(f"Embedding shape: {len(vector)} dimensions")
