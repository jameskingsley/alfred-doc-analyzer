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

# Observability Suite 
# LangGraph natively picks up LANGCHAIN_TRACING_V2 from env
try:
    from langfuse.callback import CallbackHandler
    langfuse_handler = CallbackHandler()
    logger.info("Observability: Langfuse tracing initialized.")
except Exception as e:
    langfuse_handler = None
    logger.warning(f"Observability: Langfuse tracing disabled: {e}")

@app.post("/chat")
async def chat(message: str = Form(...), file: UploadFile = File(None)):
    file_path = None
    
    # Handle File Intake
    if file:
        try:
            # Using a safe path join to avoid directory traversal issues
            safe_filename = os.path.basename(file.filename)
            file_path = os.path.join(UPLOAD_DIR, safe_filename)
            
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            logger.info(f"Alfred successfully cached file: {file_path}")
        except Exception as e:
            logger.error(f"Intake Error: {e}")
            raise HTTPException(status_code=500, detail=f"File upload failed: {str(e)}")

    # Prepare Intelligence Inputs
    inputs = {
        "messages": [HumanMessage(content=message)],
        "input_file": file_path,
        "next_agent": None, 
        "review_count": 0
    }
    
    # Execution Configuration (Tracing & Handlers)
    config = {"callbacks": [langfuse_handler]} if langfuse_handler else {}
    
    try:
        # Orchestrate Agents
        result = await react_graph.ainvoke(inputs, config=config)
        
        # Extract the Final Intelligence Report
        final_message = result["messages"][-1].content
        
        return {"response": final_message}

    except Exception as e:
        logger.error(f"Intelligence Orchestration Error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Alfred encountered a logic error: {str(e)}")

@app.get("/")
async def root():
    return {
        "status": "Online",
        "system": "Alfred Multi-Agent Intelligence",
        "observability": "Active" if langfuse_handler else "Limited"
    }