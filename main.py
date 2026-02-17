import os
from fastapi import FastAPI, UploadFile, Form, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware

from services import extract_text_from_pdf, analyze_career_gap

app = FastAPI(title="AI Career Gap Analyzer")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static & Templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/api/analyze")
async def analyze_gap(
    resume: UploadFile,
    job_title: str = Form(...)
):
    # 1. Extract Resume Text
    text = extract_text_from_pdf(resume.file)
    if not text:
        return JSONResponse(status_code=400, content={"error": "Could not read PDF text."})

    # 2. Analyze with AI
    analysis_result = analyze_career_gap(text, job_title)
    
    return analysis_result

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)

# Trigger reload for model update
