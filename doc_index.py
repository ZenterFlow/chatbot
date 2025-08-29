from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    Settings,
    StorageContext,
)
from llama_index.core.tools import QueryEngineTool, FunctionTool
from llama_index.core.agent import ReActAgent
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
import os
import chromadb
from duckduckgo_search import DDGS # We'll use this for a web search tool

# It's good practice to set your LLM and embedding models explicitly.
# This requires your OPENAI_API_KEY to be set in your environment.
Settings.llm = OpenAI(model="gpt-3.5-turbo", temperature=0.2)
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

# --- Tool Setup ---

# 1. A tool for querying your local documents (the RAG part)
query_engine_tool = QueryEngineTool.from_defaults(
    query_engine=index.as_query_engine(),
    name="document_retriever",
    description="Use this tool to retrieve information from the user's uploaded documents."
)

# 2. A tool for searching the web
def web_search(query: str) -> str:
    """Performs a web search using DuckDuckGo and returns the top 3 results."""
    with DDGS() as ddgs:
        results = [r["body"] for r in ddgs.text(query, max_results=3)]
        return "\n".join(results) if results else "No results found."

web_search_tool = FunctionTool.from_defaults(fn=web_search, name="web_search", description="Use this to search the web for recent events or topics not in the local documents.")

# --- Agent Setup ---
# An Agent is a more advanced reasoning loop that can use tools.
# The ReActAgent is a common and powerful type of agent.
agent = ReActAgent.from_tools(
    tools=[query_engine_tool, web_search_tool],
    llm=Settings.llm,
    verbose=True
)

if __name__ == "__main__":
    print("Agent is ready. It can search your documents and the web.")
    print("Type 'exit' or 'quit' to stop. Type 'reset' to clear the conversation history.")
    while True:
        question = input("Ask a question: ")
        if question.lower() in ["exit", "quit", "stop"]:
            break
        if question.lower() == "reset":
            agent.reset()
            print("Conversation history has been cleared.\n")
            continue

        response = agent.chat(question)
        print(f"Answer: {response}\n")
