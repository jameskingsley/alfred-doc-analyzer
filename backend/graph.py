import os
import fitz  
import pandas as pd
from docx import Document
from typing import TypedDict, Annotated, List, Literal, Optional
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage
from langgraph.graph import START, END, StateGraph
from langgraph.graph.message import add_messages
from dotenv import load_dotenv

load_dotenv()

# GLOBAL INITIALIZATION (Lazy Loaded for Production)
_embeddings = None
_vector_db = None

def get_vector_db():
    """Lazy loader to prevent startup timeouts on Render."""
    global _embeddings, _vector_db
    if _vector_db is None:
        if _embeddings is None:
            # Only loads when the first request is made
            _embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        
        # Using absolute path for stable container performance
        db_path = os.path.join(os.getcwd(), "vector_db")
        _vector_db = Chroma(persist_directory=db_path, embedding_function=_embeddings)
    return _vector_db

# LLM stays global (lightweight API wrapper)
llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)

# STATE DEFINITION
class AgentState(TypedDict):
    input_file: Optional[str]
    messages: Annotated[list[AnyMessage], add_messages]
    next_agent: Optional[str]
    context_data: Optional[str]  
    review_count: int            

# SPECIALIZED TOOLS

def extract_all_formats(file_path: str) -> str:
    if not file_path or not os.path.exists(file_path):
        return ""
    ext = os.path.splitext(file_path)[1].lower()
    text = ""
    try:
        if ext == ".pdf":
            text = "".join([page.get_text() for page in fitz.open(file_path)])
        elif ext in [".docx", ".doc"]:
            doc = Document(file_path)
            text = "\n".join([para.text for para in doc.paragraphs])
        elif ext in [".csv", ".xlsx"]:
            df = pd.read_csv(file_path) if ext == ".csv" else pd.read_excel(file_path)
            text = df.head(50).to_markdown() 
        elif ext in [".txt", ".md"]:
            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read()
    except Exception as e:
        return f"Error extracting text: {str(e)}"
    return text

def archive_document(file_path: str):
    content = extract_all_formats(file_path)
    if not content or "Error" in content: return
    
    vdb = get_vector_db()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = splitter.split_text(content)
    vdb.add_texts(texts=chunks, metadatas=[{"source": file_path}] * len(chunks))

#  AGENT NODES

def supervisor_node(state: AgentState):
    # Only archive if a file is present and it's the start of a thread
    if state.get("input_file") and not state.get("messages"):
        archive_document(state["input_file"])

    vdb = get_vector_db()
    query = state["messages"][-1].content
    docs = vdb.similarity_search(query, k=5) 
    memory_context = "\n---\n".join([d.page_content for d in docs])

    # Improved safe-get for input_file
    file_name = str(state.get('input_file', '')).lower()
    if any(keyword in file_name for keyword in ["resume", "cv", "kingsley"]):
        dest = "resume_specialist"
    else:
        dest = "document_analyst"
        
    return {"next_agent": dest, "context_data": memory_context, "review_count": 0}

def resume_specialist(state: AgentState):
    current_content = extract_all_formats(state["input_file"])
    prompt = (
        "You are a Senior Recruiter. Analyze the resume provided below. "
        "Use the ARCHIVED MEMORY to add context about the candidate's history. "
        f"\n\n[RESUME CONTENT]:\n{current_content}"
        f"\n\n[ARCHIVED MEMORY]:\n{state['context_data']}"
    )
    response = llm.invoke([SystemMessage(content=prompt)] + state["messages"])
    return {"messages": [response], "next_agent": "reviewer"}

def document_analyst(state: AgentState):
    current_content = extract_all_formats(state["input_file"])
    prompt = (
        "You are a Senior Data Analyst. Analyze this document meticulously. "
        "Reference data points and relate findings to the archives."
        f"\n\n[DOCUMENT CONTENT]:\n{current_content}"
        f"\n\n[ARCHIVED MEMORY]:\n{state['context_data']}"
    )
    response = llm.invoke([SystemMessage(content=prompt)] + state["messages"])
    return {"messages": [response], "next_agent": "reviewer"}

def reviewer_node(state: AgentState):
    last_report = state["messages"][-1].content
    review_prompt = (
        "You are Alfred, the Reviewer. Check for accuracy. "
        "Reply 'APPROVED' if correct, otherwise specify corrections. "
        f"\n\n[REPORT]:\n{last_report}"
    )
    review_decision = llm.invoke([SystemMessage(content=review_prompt)])
    
    if "APPROVED" in review_decision.content or state["review_count"] >= 1:
        return {"next_agent": "END"}
    
    return {"next_agent": "document_analyst", "review_count": state["review_count"] + 1}

# GRAPH CONSTRUCTION

builder = StateGraph(AgentState)
builder.add_node("supervisor", supervisor_node)
builder.add_node("resume_specialist", resume_specialist)
builder.add_node("document_analyst", document_analyst)
builder.add_node("reviewer", reviewer_node)

builder.add_edge(START, "supervisor")
builder.add_conditional_edges("supervisor", lambda x: x["next_agent"], 
    {"resume_specialist": "resume_specialist", "document_analyst": "document_analyst"})

builder.add_edge("resume_specialist", "reviewer")
builder.add_edge("document_analyst", "reviewer")

builder.add_conditional_edges("reviewer", lambda x: x["next_agent"], 
    {"document_analyst": "document_analyst", "END": END})

react_graph = builder.compile()