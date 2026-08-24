import os
import tempfile
import shutil
import dotenv
from typing import Optional
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# Import existing core agentic router
from src.agent.router import AgenticRouter

dotenv.load_dotenv()

# Top-level ASGI app object detected automatically by Vercel Functions
app = FastAPI(
    title="Agentic Video Search & Intelligence API",
    description="Multimodal Video Search Engine running on Vercel Serverless Functions",
    version="2.0.0"
)

# Enable CORS for frontend requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Lazy instantiation of AgenticRouter
router_instance: Optional[AgenticRouter] = None

def get_router() -> AgenticRouter:
    global router_instance
    if router_instance is None:
        router_instance = AgenticRouter()
    return router_instance

@app.get("/")
@app.get("/api")
@app.get("/api/health")
def health_check():
    return {
        "status": "online",
        "service": "Agentic Video Search & Intelligence API",
        "version": "2.0.0",
        "vercel_entrypoint": "api/index.py:app"
    }

@app.post("/api/analyze")
async def analyze_video(
    query: str = Form(...),
    mode: str = Form("zero_llm"),
    file: UploadFile = File(...)
):
    """Analyze uploaded video segment against a natural language query."""
    if not file.filename:
        raise HTTPException(status_code=400, detail="No video file provided.")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    try:
        r = get_router()
        res = r.answer_query(tmp_path, query, mode=mode)
        return JSONResponse(content=res)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception:
                pass

@app.post("/api/summary")
async def generate_summary(
    mode: str = Form("zero_llm"),
    file: UploadFile = File(...)
):
    """Generate executive summary narrative and chronological window log for uploaded video."""
    if not file.filename:
        raise HTTPException(status_code=400, detail="No video file provided.")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    try:
        r = get_router()
        res = r.generate_full_video_log(tmp_path, mode=mode)
        return JSONResponse(content=res)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception:
                pass
