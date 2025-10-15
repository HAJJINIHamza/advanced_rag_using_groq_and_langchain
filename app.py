#Imports 
import os 
from dotenv import load_dotenv
import faiss
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from typing_extensions import TypedDict, List
from langchain.prompts import PromptTemplate
from langgraph.graph import StateGraph, START
import streamlit as st
import pdfplumber

#Load environement variables
load_dotenv()

#LLM
groq_llm = ChatGroq(
    groq_api_key = os.environ["GROQ_API_KEY"],
    model="llama-3.1-8b-instant",
    temperature = 0.5
)

#Embedder 
embedder = HuggingFaceEmbeddings(
    model_name ="sentence-transformers/all-MiniLM-L6-v2"
)

#Vector database 
embeddings_ex = embedder.embed_query("hi")
index = faiss.IndexFlatL2(len(embeddings_ex))
vectore_store = FAISS(
    embedding_function = embedder,
    index=index,
    docstore=InMemoryDocstore(),
    index_to_docstore_id={}
)

#Load Documents 
documents = []
with pdfplumber.open("documents/I-Will-Teach-You-to-Be-Rich-Book-Summary.pdf") as book:
    for page in book.pages:
        documents.append(page.extract_text())
documents = "\n".join(documents)
#make it document 
documents_structured = Document(page_content=documents)

#Splitting 
splitter = RecursiveCharacterTextSplitter(chunk_size = 500, chunk_overlap = 100)
documents_splitted = splitter.split_documents( [documents_structured])

#Add documents to our vectore store 
_ = vectore_store.add_documents(documents_splitted)

#Define our app input and output types 
class State(TypedDict):
    question : str
    context : List
    answer : str

#Retrieve funct
def retrieve(state : State):
    retrieved_docs = vectore_store.similarity_search(state["question"]) 
    return {"context": retrieved_docs}

#Define prompt
prompt = """
    Use relavant informations bellow, to answer the following question : 
    {question}
    If you don't know the answer, don't make up one, simply say I don't know.
    relevant informations : 
    {context}
"""
prompt = PromptTemplate(
    template = prompt,
    input_variables = ["question", "context"]
    )

#Generate funct
def generate(state : State):
    context = "\n\n relevant information :\n".join(text.page_content for text in state["context"])
    messages = prompt.invoke({"question": state["question"], "context": context})
    response = groq_llm.invoke(messages)
    return {"answer": response}

#Build and compile application 
workflow = StateGraph(State).add_sequence([retrieve, generate])
workflow.add_edge(START, "retrieve")
rag_app = workflow.compile()

#Streamlit app
st.set_page_config(page_title="Chat with RAG, powered by Groq capabilities.")
st.header("Chat with RAG, powered by Groq capabilites.")

if question := st.chat_input("Ask a question"):
    st.chat_message("User").write(question)
    #Include streaming 
    with st.chat_message("assistant"):
        st.write_stream(
            message.content for message, _ in rag_app.stream(
                {"question": question},
                stream_mode = 'messages'
            )
        )   


#ADD streaming
#ADD sources
#Give the user option to include its own documents 
#Make it conversational style 


