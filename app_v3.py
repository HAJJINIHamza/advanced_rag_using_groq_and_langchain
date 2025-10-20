###############################################################################################################################################################
###########################################                                                                       #############################################
########################################### ADVANCED RAG THAT SUPPORTS CONVERSATIONAL STYLE AND ADVANCED FEATURES #############################################
###########################################                                                                       #############################################
###############################################################################################################################################################

import os 
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langgraph.graph import MessagesState, StateGraph, END
from langchain_core.messages import SystemMessage
from langchain.tools import tool
from langgraph.prebuilt import ToolNode, tools_condition
import streamlit as st
import pdfplumber

#
load_dotenv()

#
st.set_page_config("Q&A GROQ RAG")
st.header("Chat with Advanced Q&A_GROQ_RAG")

#LLM
groq_llm = ChatGroq(
    groq_api_key = os.environ["GROQ_API_KEY"],
    model="llama-3.1-8b-instant",
    temperature = 0.6
)
#Embdedder
embedder = HuggingFaceEmbeddings(
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
) 
#Read pdf documents 
def read_pdf_file(pdf_file):
    #Read document using pdfplumber
    documents = []
    with pdfplumber.open(pdf_file) as book:
        for page in book.pages:
            documents.append(page.extract_text())
    documents = "\n".join(documents)
    return Document(documents)


#Vector store
def get_vector_store(pdf_file, verbose = False):
    if "vector_store" not in st.session_state:
        if verbose == True:
            print ("No vector store in session state, create new one.")
        st.session_state.vector_store = InMemoryVectorStore(embedding=embedder)
        documents = read_pdf_file(pdf_file)
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap = 200)
        chunks = splitter.split_documents([documents])
        st.session_state.vector_store.add_documents(chunks)
        if verbose == True:
            print ("Chunked, embedded and added documents to vector store")
        return len(chunks)

    else :
        if verbose == True:
            print("Vector store already exists, return it.")
        return 0
        

#Building graph components 
#Tool
@tool(response_format="content_and_artifact")
def retrieve(query:str):
    """Function to retrieve relevant document for query"""
    vec_store = st.session_state.vector_store
    retrieved_documents = vec_store.similarity_search(query, k=3)
    summary = "\n\n".join(
        [f"Source : \n {doc.metadata} \nContent: \n {doc.page_content}" for doc in retrieved_documents]
    )
    return summary, retrieved_documents

#Node 1 
def query_or_respond(state: MessagesState):
    """LLM should either respond directly or rewrite query for retrieval"""
    groq_llm_with_tools = groq_llm.bind_tools([retrieve])
    response = groq_llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}

#Node 2
tools = ToolNode([retrieve])

#Node 3 : generate
def generate(state: MessagesState):
    """Generate answer to query"""
    #Extract tool messages et ensuite retrieved docs
    recent_tool_messages = []
    for message in reversed(state["messages"]):
        if message.type == "tool":
            recent_tool_messages.append(message)
        else:
            break
    retrieved_docs = recent_tool_messages[::-1]
    docs_content = "\n\n".join([doc.content for doc in retrieved_docs])

    #Query + chat_history
    chat_history = []
    for message in state["messages"]:
        if ((message.type in ["humna", "system"]) or 
        (message.type == "ai" and not message.tool_calls)):
            chat_history.append(message)

    #Prompt
    system_prompt = """
        You are a Q&A assistant.
        Answer questions using relevant docs bellow, 
        If you don't know the answer just say so, don't make one.
        Relevant informatons :
        {relevant_docs}  
    """

    #Pompt finale
    prompt_finale = [SystemMessage(system_prompt)] + chat_history
    response = groq_llm.invoke(prompt_finale)
    return {"message": [response]}

#Build final graph
workflow = StateGraph(MessagesState)
#Include nodes
workflow.add_node(query_or_respond)
workflow.add_node(tools)
workflow.add_node(generate)
#Include edges
workflow.set_entry_point("query_or_respond")
workflow.add_conditional_edges(
    "query_or_respond",
    tools_condition,
    {END:END, "tools": "tools"}
)
workflow.add_edge("tools", "generate")
workflow.add_edge("generate", END)
#Compile
rag_app = workflow.compile()

#Streamlit 
#Ask user to upload a pdf file
pdf_file = st.file_uploader(
    "Upload your files here",
    type = ["pdf"]
)
#Process pdf file and store it in vector store
if pdf_file is not None:
    if "processed_files" not in st.session_state:
        st.session_state.processed_files = set()
    
    if pdf_file.name not in st.session_state.processed_file: 
        with st.spinner("Processing file ..."):
            #Create Vector store
            num_chunks = get_vector_store(pdf_file, verbose = True)
            if num_chunks > 0:
                st.session_state.processed_files.append(pdf_file.name)
                st.success(f"✅ Processed {pdf_file.name} into {num_chunks} chunks")

#Answer user question
if question := st.chat_input:
    input = {"messages": 
             [{"role": "user", "content": question}]
             }
    response = rag_app.invoke(input)
    final_response = response["messages"][-1].content
    st.chat_message("user").write(question)
    st.chat_message("assistant").write(final_response)

# Add a clear button to reset the session
with st.sidebar:
    if st.button("Clear chat history"):
        st.session_state.clear()
        st.rerun()










    



