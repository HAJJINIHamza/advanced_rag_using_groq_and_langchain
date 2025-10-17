import streamlit as st
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langgraph.graph import StateGraph, START
from typing import TypedDict, List, Annotated
from langchain_community.docstore.in_memory import InMemoryDocstore
import faiss
import pdfplumber
import os
from dotenv import load_dotenv

# load variables
load_dotenv()

# Streamlit app
st.set_page_config(page_title="Chat with RAG, powered by Groq capabilities.")
st.header("Chat with RAG, powered by Groq capabilities.")

# Initialize LLM
groq_llm = ChatGroq(
    groq_api_key=os.environ["GROQ_API_KEY"],
    model="llama-3.1-8b-instant",
    temperature=0.5
)

# Initialize embedder
embedder = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    cache_folder="/tmp/models"
)

# Initialize vector store in session state
def get_vector_store():
    """Get or create vector store for this session"""
    if "vector_store" not in st.session_state:
        embeddings_ex = embedder.embed_query("hi")
        index = faiss.IndexFlatL2(len(embeddings_ex))
        st.session_state.vector_store = FAISS(
            embedding_function=embedder,
            index=index,
            docstore=InMemoryDocstore(),
            index_to_docstore_id={}
        )
    return st.session_state.vector_store

# Function to handle pdf file
def read_structure_chunk_and_store_pdf(pdf_file):
    """Process PDF and add to user's session vector store"""
    vector_store = get_vector_store()
    
    # Load Documents
    documents = []
    with pdfplumber.open(pdf_file) as book:
        for page in book.pages:
            text = page.extract_text()
            if text:
                documents.append(text)
    
    if not documents:
        st.warning("No text could be extracted from the PDF")
        return 0
    
    documents_text = "\n".join(documents)
    
    # Make it document
    documents_structured = Document(page_content=documents_text)
    
    # Splitting
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    documents_splitted = splitter.split_documents([documents_structured])
    
    # Add documents to our vector store
    vector_store.add_documents(documents_splitted)
    
    return len(documents_splitted)

# Define our app input and output types - INCLUDE vector_store in state
class State(TypedDict):
    question: str
    context: List
    answer: str
    vector_store: FAISS  # Pass vector store through the state!

# Retrieve function - use vector_store from state
def retrieve(state: State):
    """Retrieve relevant documents using the vector store from state"""
    vector_store = state["vector_store"]
    retrieved_docs = vector_store.similarity_search(state["question"], k=4)
    return {"context": retrieved_docs}

# Define prompt
prompt_template = """
Use relevant informations below, to answer the following question: 
{question}
If you don't know the answer, don't make up one, simply say I don't know.

Relevant informations: 
{context}
"""
prompt = PromptTemplate(
    template=prompt_template,
    input_variables=["question", "context"]
)

# Generate function
def generate(state: State):
    """Generate answer using retrieved context"""
    if not state["context"]:
        return {"answer": "I don't have any relevant information to answer your question. Please upload a PDF document first."}
    
    context = "\n\n".join(text.page_content for text in state["context"])
    messages = prompt.invoke({"question": state["question"], "context": context})
    response = groq_llm.invoke(messages)
    return {"answer": response}

# Build and compile application
workflow = StateGraph(State).add_sequence([retrieve, generate])
workflow.add_edge(START, "retrieve")
rag_app = workflow.compile()

# Show number of documents in vector store
if "vector_store" in st.session_state:
    num_docs = len(st.session_state.vector_store.docstore._dict)
    if num_docs > 0:
        st.info(f"📚 Documents in memory: {num_docs} chunks")

uploaded_file = st.file_uploader(
    "Upload document",
    type=["pdf"]
)

if uploaded_file is not None:
    if "processed_files" not in st.session_state:
        st.session_state.processed_files = set()
    
    # Only process if not already processed in this session
    if uploaded_file.name not in st.session_state.processed_files:
        with st.spinner("Processing PDF..."):
            num_chunks = read_structure_chunk_and_store_pdf(pdf_file=uploaded_file)
            if num_chunks > 0:
                st.session_state.processed_files.add(uploaded_file.name)
                st.success(f"✅ Processed {uploaded_file.name} into {num_chunks} chunks")

if question := st.chat_input("Ask a question"):
    # Check if any documents have been uploaded
    if "vector_store" not in st.session_state or len(st.session_state.vector_store.docstore._dict) == 0:
        st.warning("⚠️ Please upload a PDF document first!")
    else:
        st.chat_message("user").write(question)
        
        # Get the vector store from session state
        vector_store = st.session_state.vector_store
        
        # Include streaming - PASS vector_store in the initial state
        with st.chat_message("assistant"):
            st.write_stream(
                message.content for message, _ in rag_app.stream(
                    {
                        "question": question,
                        "vector_store": vector_store  # Pass it here!
                    },
                    stream_mode='messages'
                )
            )

# Add a clear button to reset the session
with st.sidebar:
    if st.button("Clear chat history"):
        st.session_state.clear()
        st.rerun()