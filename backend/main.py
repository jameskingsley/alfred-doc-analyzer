from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from backend.graph import react_graph
from langchain_core.messages import HumanMessage
import shutil
import os
import logging

# Initialize Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Alfred: Multi-Agent API")

# Ensure the assets directory exists for processing
UPLOAD_DIR = "assets"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Optional Langfuse Integration
try:
    from langfuse.langchain import CallbackHandler
    langfuse_handler = CallbackHandler()
    logger.info("Langfuse tracing initialized.")
except Exception as e:
    langfuse_handler = None
    logger.warning(f"Langfuse tracing disabled or unauthorized: {e}")

@app.post("/chat")
async def chat(message: str = Form(...), file: UploadFile = File(None)):
    file_path = None
    
    # Handle File Upload
    if file:
        try:
            file_path = os.path.join(UPLOAD_DIR, file.filename)
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            logger.info(f"File stored at: {file_path}")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"File upload failed: {str(e)}")

    # Prepare inputs for the Multi-Agent Orchestrator
    inputs = {
        "messages": [HumanMessage(content=message)],
        "input_file": file_path,
        "next_agent": "supervisor" 
    }
    
    # Execution Config 
    config = {"callbacks": [langfuse_handler]} if langfuse_handler else {}
    
    try:
        # Invoke the Multi-Agent Graph
        # Using ainvoke for async performance in FastAPI
        result = await react_graph.ainvoke(inputs, config=config)
        
        # Extract Alfred's final response
        final_message = result["messages"][-1].content
        
        return {"response": final_message}

    except Exception as e:
        logger.error(f"Graph Execution Error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Alfred encountered an error: {str(e)}")

# Root endpoint for health checks
@app.get("/")
async def root():
    return {"status": "Alfred is online and awaiting Master Wayne's instructions."}