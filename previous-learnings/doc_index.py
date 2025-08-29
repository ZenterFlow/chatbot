from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    Settings,
    StorageContext,
)
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
import os
import chromadb

# It's good practice to set your LLM and embedding models explicitly.
# This requires your OPENAI_API_KEY to be set in your environment.
Settings.llm = OpenAI(model="gpt-3.5-turbo")
Settings.embed_model = OpenAIEmbedding(model="text-embedding-ada-002")

DATA_FOLDER = "data"
CHROMA_PATH = "./chroma_db"
COLLECTION_NAME = "my_docs"

# Function to load or build index
def load_index():
    # initialize client and collection
    db = chromadb.PersistentClient(path=CHROMA_PATH)
    chroma_collection = db.get_or_create_collection(COLLECTION_NAME)

    # assign chroma as the vector_store to the context
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # check if the collection is empty to decide whether to build or load
    if chroma_collection.count() == 0:
        print("Building index from documents and storing in ChromaDB...")
        # load documents
        docs = SimpleDirectoryReader(DATA_FOLDER).load_data()
        # create the index, which will also store it in ChromaDB
        index = VectorStoreIndex.from_documents(docs, storage_context=storage_context)
        print("Index built and stored.")
    else:
        print("Loading index from ChromaDB...")
        # load the index from the existing vector store
        index = VectorStoreIndex.from_vector_store(
            vector_store, storage_context=storage_context
        )
        print("Index loaded.")

    return index

# Load the index (from ChromaDB or build new)
index = load_index()

# Query function
def query_index(question):
    query_engine = index.as_query_engine()
    response = query_engine.query(question)
    return str(response)

if __name__ == "__main__":
    print("Chatbot is ready. Type 'exit' or 'quit' to stop.")
    while True:
        question = input("Ask a question: ")
        if question.lower() in ["exit", "quit"]:
            break
        answer = query_index(question)
        print(f"Answer: {answer}\n")
