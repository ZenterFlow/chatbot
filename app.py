import streamlit as st
import os
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain_community.document_loaders import PyPDFLoader, TextLoader, UnstructuredMarkdownLoader
import chromadb

# --- App Configuration ---
st.set_page_config(page_title="Chat with your Docs", layout="centered")
st.title("📄 Chat with your Documents")
st.caption("Powered by LangChain, OpenAI, and ChromaDB")

# --- Constants ---
DATA_FOLDER = "data"
CHROMA_PATH = "./chroma_db"
COLLECTION_NAME = "my_docs"

# Ensure the data directory exists
if not os.path.exists(DATA_FOLDER):
    os.makedirs(DATA_FOLDER)

# --- Model and Embedding Setup ---
# This requires your OPENAI_API_KEY to be set in your environment.
# You can also set it directly in Streamlit secrets.
try:
    llm = ChatOpenAI(model="gpt-3.5-turbo", streaming=True)
    embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
except Exception as e:
    st.error(f"Error setting up OpenAI models. Have you set your OPENAI_API_KEY? Error: {e}")
    st.stop()


# --- Vector Store and Chain Setup ---
@st.cache_resource(show_spinner="Loading vector store and creating chain...")
def get_retrieval_chain_and_vector_store():
    """
    Loads the vector store from ChromaDB and creates a conversational retrieval chain.
    The function is cached to avoid reloading on every interaction.
    """
    db = chromadb.PersistentClient(path=CHROMA_PATH)
    vector_store = Chroma(
        client=db,
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
    )

    # Create a conversational retrieval chain
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vector_store.as_retriever(),
        return_source_documents=True,
    )
    return chain, vector_store

chain, vector_store = get_retrieval_chain_and_vector_store()

# --- Sidebar for File Upload ---
with st.sidebar:
    st.header("📚 Add to Knowledge Base")
    uploaded_file = st.file_uploader(
        "Upload a document (.pdf, .txt, .md)",
        type=["pdf", "txt", "md"],
        help="Upload a document to add its contents to the knowledge base."
    )

    if uploaded_file:
        file_path = ""
        try:
            # Save the uploaded file to the data directory
            file_path = os.path.join(DATA_FOLDER, uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getvalue())

            with st.spinner(f"Reading and indexing '{uploaded_file.name}'..."):
                # Determine loader based on file type
                ext = os.path.splitext(uploaded_file.name)[1].lower()
                if ext == ".pdf":
                    loader = PyPDFLoader(file_path)
                elif ext == ".txt":
                    loader = TextLoader(file_path)
                elif ext == ".md":
                    loader = UnstructuredMarkdownLoader(file_path)
                else:
                    st.error(f"Unsupported file type: {ext}")
                    st.stop()

                documents = loader.load()

                # Split documents into chunks
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000, chunk_overlap=200
                )
                chunks = text_splitter.split_documents(documents)

                # Add chunks to the vector store
                vector_store.add_documents(chunks)

            st.success(f"✅ Successfully indexed '{uploaded_file.name}'!")
            st.info("You can now ask questions about the new document.")

        except Exception as e:
            st.error(f"❌ An error occurred: {e}")
            # Clean up the saved file if indexing fails
            if file_path and os.path.exists(file_path):
                os.remove(file_path)


# --- Chat Interface ---
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! Ask me anything about your documents."}
    ]
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []  # For LangChain

# Display chat messages from history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("Ask a question..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Get response from the query engine
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            # The chain expects a dictionary with "question" and "chat_history"
            result = chain(
                {"question": prompt, "chat_history": st.session_state.chat_history}
            )
            response_content = result["answer"]
            st.markdown(response_content)

            # Update histories
            st.session_state.chat_history.append((prompt, response_content))
            st.session_state.messages.append(
                {"role": "assistant", "content": response_content}
            )
