import os
import sys
import shutil
import json
import tempfile

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
from fastapi.responses import FileResponse, Response, JSONResponse
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

# Use temporary directory guaranteed to be writable on cloud platforms (Render/Linux)
UPLOAD_DIR = os.path.join(tempfile.gettempdir(), "insightstream_uploads")
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
        return JSONResponse(status_code=500, content={"detail": f"Failed to save settings: {e}"})

class QueryRequest(BaseModel):
    query: str
    filename: Optional[str] = "all"
    mode: Optional[str] = "structured"

def get_df(filename: str) -> pd.DataFrame:
    filename_clean = os.path.basename(filename)
    if filename_clean in DATASETS_CACHE:
        return DATASETS_CACHE[filename_clean]
    filepath = os.path.join(UPLOAD_DIR, filename_clean)
    if not os.path.exists(filepath):
        raise HTTPException(status_code=404, detail="File not found")
    try:
        df = pd.read_csv(filepath)
        DATASETS_CACHE[filename_clean] = df
        return df
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read CSV: {e}")

@app.post("/api/upload")
async def upload_file(file: UploadFile = File(...)):
    try:
        if not file.filename.endswith(".csv"):
            return JSONResponse(status_code=400, content={"detail": "Only CSV files are supported."})
        
        filename = os.path.basename(file.filename)
        filepath = os.path.join(UPLOAD_DIR, filename)
        
        try:
            with open(filepath, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
        except Exception as save_err:
            return JSONResponse(status_code=500, content={"detail": f"Failed to save file: {save_err}"})
            
        try:
            df = pd.read_csv(filepath)
            DATASETS_CACHE[filename] = df
        except Exception as csv_err:
            if os.path.exists(filepath):
                os.remove(filepath)
            return JSONResponse(status_code=400, content={"detail": f"Invalid CSV structure: {csv_err}"})
            
        num_rows, num_cols = df.shape
        columns_profile = []
        
        for col in df.columns:
            series = df[col]
            null_count = int(series.isnull().sum())
            unique_count = int(series.nunique())
            dtype = str(series.dtype)
            
            col_info = {
                "name": str(col),
                "type": dtype,
                "null_count": null_count,
                "null_percentage": round((null_count / num_rows) * 100, 2) if num_rows > 0 else 0,
                "unique_count": unique_count,
                "sample_values": [str(v) for v in series.dropna().head(5).tolist()]
            }
            
            if np.issubdtype(series.dtype, np.number):
                try:
                    mean_val = float(series.mean()) if not pd.isnull(series.mean()) else None
                    min_val = float(series.min()) if not pd.isnull(series.min()) else None
                    max_val = float(series.max()) if not pd.isnull(series.max()) else None
                    std_val = float(series.std()) if not pd.isnull(series.std()) else None
                    
                    col_info.update({
                        "mean": round(mean_val, 2) if mean_val is not None else None,
                        "min": round(min_val, 2) if min_val is not None else None,
                        "max": round(max_val, 2) if max_val is not None else None,
                        "std": round(std_val, 2) if std_val is not None else None,
                        "is_numeric": True
                    })
                except Exception:
                    col_info.update({"is_numeric": False})
            else:
                try:
                    top_counts = series.value_counts().head(5)
                    col_info.update({
                        "top_values": [{"val": str(k), "count": int(v)} for k, v in top_counts.items()],
                        "is_numeric": False
                    })
                except Exception:
                    col_info.update({"top_values": [], "is_numeric": False})
                
            columns_profile.append(col_info)
            
        try:
            sample_rows = json.loads(df.head(10).to_json(orient="records"))
        except Exception:
            sample_rows = []
            
        return JSONResponse(status_code=200, content={
            "filename": filename,
            "rows": num_rows,
            "columns_count": num_cols,
            "columns": columns_profile,
            "preview": sample_rows,
            "message": "File uploaded and profiled successfully."
        })
        
    except Exception as general_err:
        return JSONResponse(status_code=500, content={"detail": f"Upload process failed: {str(general_err)}"})

@app.get("/api/files/{filename}")
async def get_file_profile(filename: str):
    try:
        df = get_df(filename)
        num_rows, num_cols = df.shape
        columns_profile = []
        for col in df.columns:
            series = df[col]
            null_count = int(series.isnull().sum())
            unique_count = int(series.nunique())
            dtype = str(series.dtype)
            col_info = {
                "name": str(col),
                "type": dtype,
                "null_count": null_count,
                "null_percentage": round((null_count / num_rows) * 100, 2) if num_rows > 0 else 0,
                "unique_count": unique_count,
                "sample_values": [str(v) for v in series.dropna().head(5).tolist()]
            }
            if np.issubdtype(series.dtype, np.number):
                try:
                    mean_val = float(series.mean()) if not pd.isnull(series.mean()) else None
                    min_val = float(series.min()) if not pd.isnull(series.min()) else None
                    max_val = float(series.max()) if not pd.isnull(series.max()) else None
                    std_val = float(series.std()) if not pd.isnull(series.std()) else None
                    col_info.update({
                        "mean": round(mean_val, 2) if mean_val is not None else None,
                        "min": round(min_val, 2) if min_val is not None else None,
                        "max": round(max_val, 2) if max_val is not None else None,
                        "std": round(std_val, 2) if std_val is not None else None,
                        "is_numeric": True
                    })
                except Exception:
                    col_info.update({"is_numeric": False})
            else:
                try:
                    top_counts = series.value_counts().head(5)
                    col_info.update({
                        "top_values": [{"val": str(k), "count": int(v)} for k, v in top_counts.items()],
                        "is_numeric": False
                    })
                except Exception:
                    col_info.update({"top_values": [], "is_numeric": False})
            columns_profile.append(col_info)
        sample_rows = json.loads(df.head(10).to_json(orient="records"))
        return JSONResponse(status_code=200, content={
            "filename": filename,
            "rows": num_rows,
            "columns_count": num_cols,
            "columns": columns_profile,
            "preview": sample_rows,
            "message": "File profiled successfully."
        })
    except Exception as e:
        return JSONResponse(status_code=500, content={"detail": f"Profiling failed: {str(e)}"})

@app.post("/api/query")
async def query_dataset(request: QueryRequest):
    try:
        df = get_df(request.filename)
        result = run_insight_query(
            request.query, 
            df, 
            mode=request.mode or "structured", 
            filename=request.filename
        )
        return result
    except Exception as e:
        return JSONResponse(status_code=500, content={"detail": f"Query execution failed: {str(e)}"})

@app.get("/api/files")
async def list_files():
    files = []
    if os.path.exists(UPLOAD_DIR):
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
    filename_clean = os.path.basename(filename)
    filepath = os.path.join(UPLOAD_DIR, filename_clean)
    if os.path.exists(filepath):
        os.remove(filepath)
    if filename_clean in DATASETS_CACHE:
        del DATASETS_CACHE[filename_clean]
    return {"message": f"File '{filename_clean}' deleted successfully."}

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
        return JSONResponse(status_code=404, content={"detail": "API endpoint not found"})
        
    base_name = os.path.basename(file_path)
    
    # Check exact relative path or filename basename across all possible folders
    for folder in [os.path.join(base_dir, "dist"), os.path.join(base_dir, "frontend", "dist"), base_dir]:
        for candidate in [os.path.join(folder, file_path), os.path.join(folder, base_name)]:
            if os.path.isfile(candidate):
                return FileResponse(candidate)
                
    # SPA fallback for page navigation
    for index_path in [
        os.path.join(base_dir, "dist", "index.html"),
        os.path.join(base_dir, "frontend", "dist", "index.html"),
        os.path.join(base_dir, "index.html")
    ]:
        if os.path.exists(index_path):
            return FileResponse(index_path)
            
    return JSONResponse(status_code=404, content={"detail": "Not Found"})

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)
