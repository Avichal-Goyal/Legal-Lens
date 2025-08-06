from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import shutil
import os
import logging
from agents import read_pdf_file, get_summary, extract_clauses

app = FastAPI(title = "Legal_Lens API")

origins = ["http://localhost:3000"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ResultStructure(BaseModel):
    summary: str
    clauses: str

@app.post("/simplify_document", response_model=ResultStructure)

async def simplify_document(uploaded_file: UploadFile):
    filePath = f"temp_{uploaded_file.filename}"
    try:
        with open(filePath, "wb") as buffer:
            shutil.copyfileobj(uploaded_file.file, buffer)

        doc_text = read_pdf_file(filePath)
        summary = get_summary(doc_text)
        clauses = extract_clauses(doc_text)

        return {"summary": summary, "clauses": clauses}
    except Exception as e:
        logging.error(f"Error during document analysis: {e}")
        return {"summary": "Error", "clauses": "Error"}
    finally:
        if os.path.exists(filePath):
            os.remove(filePath)