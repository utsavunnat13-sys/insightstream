import os
import sys
import shutil

# Ensure current directory and backend directory are in sys.path
base_dir = os.path.dirname(os.path.abspath(__file__))
if base_dir not in sys.path:
    sys.path.insert(0, base_dir)
backend_dir = os.path.join(base_dir, "backend")
if os.path.exists(backend_dir) and backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

import pandas as pd
import numpy as np
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import dotenv

# Load environment variables
dotenv.load_dotenv()

from query_engine import run_insight_query
from settings import load_settings, save_settings, AppSettings, mask_key

app = FastAPI(title="InsightStream API", description="AI-powered Conversational BI API")

# Configure CORS for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory storage for active datasets
DATASETS_CACHE: Dict[str, pd.DataFrame] = {}

UPLOAD_DIR = os.path.join(base_dir, "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

class SettingsUpdatePayload(BaseModel):
    provider: str
    gemini_api_key: Optional[str] = None
    gemini_model: Optional[str] = None
    openai_api_key: Optional[str] = None
    openai_model: Optional[str] = None
    anthropic_api_key: Optional[str] = None
    anthropic_model: Optional[str] = None
    ollama_api_base: Optional[str] = None
    ollama_model: Optional[str] = None

@app.get("/api/settings")
async def get_app_settings():
    current = load_settings()
    return {
        "provider": current.provider,
        "gemini_api_key": mask_key(current.gemini_api_key) if current.gemini_api_key else None,
        "gemini_model": current.gemini_model,
        "openai_api_key": mask_key(current.openai_api_key) if current.openai_api_key else None,
        "openai_model": current.openai_model,
        "anthropic_api_key": mask_key(current.anthropic_api_key) if current.anthropic_api_key else None,
        "anthropic_model": current.anthropic_model,
        "ollama_api_base": current.ollama_api_base,
        "ollama_model": current.ollama_model
    }

@app.post("/api/settings")
async def update_app_settings(payload: SettingsUpdatePayload):
    current = load_settings()
    new_settings = AppSettings(
        provider=payload.provider,
        gemini_api_key=payload.gemini_api_key if payload.gemini_api_key is not None else current.gemini_api_key,
        gemini_model=payload.gemini_model or current.gemini_model,
        openai_api_key=payload.openai_api_key if payload.openai_api_key is not None else current.openai_api_key,
        openai_model=payload.openai_model or current.openai_model,
        anthropic_api_key=payload.anthropic_api_key if payload.anthropic_api_key is not None else current.anthropic_api_key,
        anthropic_model=payload.anthropic_model or current.anthropic_model,
        ollama_api_base=payload.ollama_api_base or current.ollama_api_base,
        ollama_model=payload.ollama_model or current.ollama_model
    )
    try:
        save_settings(new_settings)
        return {"success": True, "message": "Settings updated successfully."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save settings: {e}")

class QueryRequest(BaseModel):
    query: str
    filename: Optional[str] = "all"
    mode: Optional[str] = "structured"

def get_df(filename: str) -> pd.DataFrame:
    if filename in DATASETS_CACHE:
        return DATASETS_CACHE[filename]
    filepath = os.path.join(UPLOAD_DIR, filename)
    if not os.path.exists(filepath):
        raise HTTPException(status_code=404, detail="File not found")
    try:
        df = pd.read_csv(filepath)
        DATASETS_CACHE[filename] = df
        return df
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read CSV: {e}")

@app.post("/api/upload")
async def upload_file(file: UploadFile = File(...)):
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are supported.")
    filepath = os.path.join(UPLOAD_DIR, file.filename)
    try:
        with open(filepath, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save file: {e}")
    try:
        df = pd.read_csv(filepath)
        DATASETS_CACHE[file.filename] = df
    except Exception as e:
        if os.path.exists(filepath):
            os.remove(filepath)
        raise HTTPException(status_code=400, detail=f"Invalid CSV structure: {e}")
    num_rows, num_cols = df.shape
    columns_profile = []
    for col in df.columns:
        series = df[col]
        null_count = int(series.isnull().sum())
        unique_count = int(series.nunique())
        dtype = str(series.dtype)
        col_info = {
            "name": col,
            "type": dtype,
            "null_count": null_count,
            "null_percentage": round((null_count / num_rows) * 100, 2) if num_rows > 0 else 0,
            "unique_count": unique_count,
            "sample_values": series.dropna().head(5).tolist()
        }
        if np.issubdtype(series.dtype, np.number):
            col_info.update({
                "mean": round(float(series.mean()), 2) if not pd.isnull(series.mean()) else None,
                "min": float(series.min()) if not pd.isnull(series.min()) else None,
                "max": float(series.max()) if not pd.isnull(series.max()) else None,
                "std": round(float(series.std()), 2) if not pd.isnull(series.std()) else None,
                "is_numeric": True
            })
        else:
            top_counts = series.value_counts().head(5)
            col_info.update({
                "top_values": [{"val": str(k), "count": int(v)} for k, v in top_counts.items()],
                "is_numeric": False
            })
        columns_profile.append(col_info)
    sample_rows = df.head(10).replace([np.inf, -np.inf], np.nan).where(pd.notnull(df), None).to_dict(orient="records")
    return {
        "filename": file.filename,
        "rows": num_rows,
        "columns_count": num_cols,
        "columns": columns_profile,
        "preview": sample_rows,
        "message": "File uploaded and profiled successfully."
    }

@app.get("/api/files/{filename}")
async def get_file_profile(filename: str):
    df = get_df(filename)
    num_rows, num_cols = df.shape
    columns_profile = []
    for col in df.columns:
        series = df[col]
        null_count = int(series.isnull().sum())
        unique_count = int(series.nunique())
        dtype = str(series.dtype)
        col_info = {
            "name": col,
            "type": dtype,
            "null_count": null_count,
            "null_percentage": round((null_count / num_rows) * 100, 2) if num_rows > 0 else 0,
            "unique_count": unique_count,
            "sample_values": series.dropna().head(5).tolist()
        }
        if np.issubdtype(series.dtype, np.number):
            col_info.update({
                "mean": round(float(series.mean()), 2) if not pd.isnull(series.mean()) else None,
                "min": float(series.min()) if not pd.isnull(series.min()) else None,
                "max": float(series.max()) if not pd.isnull(series.max()) else None,
                "std": round(float(series.std()), 2) if not pd.isnull(series.std()) else None,
                "is_numeric": True
            })
        else:
            top_counts = series.value_counts().head(5)
            col_info.update({
                "top_values": [{"val": str(k), "count": int(v)} for k, v in top_counts.items()],
                "is_numeric": False
            })
        columns_profile.append(col_info)
    sample_rows = df.head(10).replace([np.inf, -np.inf], np.nan).where(pd.notnull(df), None).to_dict(orient="records")
    return {
        "filename": filename,
        "rows": num_rows,
        "columns_count": num_cols,
        "columns": columns_profile,
        "preview": sample_rows,
        "message": "File profiled successfully."
    }

@app.post("/api/query")
async def query_dataset(request: QueryRequest):
    df = get_df(request.filename)
    result = run_insight_query(
        request.query, 
        df, 
        mode=request.mode or "structured", 
        filename=request.filename
    )
    return result

@app.get("/api/files")
async def list_files():
    files = []
    for f in os.listdir(UPLOAD_DIR):
        if f.endswith(".csv"):
            filepath = os.path.join(UPLOAD_DIR, f)
            size = os.path.getsize(filepath)
            files.append({
                "filename": f,
                "size_bytes": size,
                "cached": f in DATASETS_CACHE
            })
    return files

@app.delete("/api/files/{filename}")
async def delete_file(filename: str):
    filepath = os.path.join(UPLOAD_DIR, filename)
    if os.path.exists(filepath):
        os.remove(filepath)
    if filename in DATASETS_CACHE:
        del DATASETS_CACHE[filename]
    return {"message": f"File '{filename}' deleted successfully."}

# Static assets and SPA HTML fallback
@app.get("/")
async def serve_root():
    for path in [
        os.path.join(base_dir, "dist", "index.html"),
        os.path.join(base_dir, "frontend", "dist", "index.html"),
        os.path.join(base_dir, "index.html")
    ]:
        if os.path.exists(path):
            return FileResponse(path)
    return Response(content="<h1>InsightStream API is Active</h1>", media_type="text/html")

@app.get("/{file_path:path}")
async def serve_static_or_spa(file_path: str):
    if file_path.startswith("api"):
        raise HTTPException(status_code=404, detail="API endpoint not found")
    for folder in [os.path.join(base_dir, "dist"), os.path.join(base_dir, "frontend", "dist"), base_dir]:
        target = os.path.join(folder, file_path)
        if os.path.isfile(target):
            return FileResponse(target)
    for index_path in [
        os.path.join(base_dir, "dist", "index.html"),
        os.path.join(base_dir, "frontend", "dist", "index.html"),
        os.path.join(base_dir, "index.html")
    ]:
        if os.path.exists(index_path):
            return FileResponse(index_path)
    raise HTTPException(status_code=404, detail="Not Found")

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)
