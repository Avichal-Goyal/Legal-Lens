import os
import shutil
import logging
import json
from dotenv import load_dotenv
import google.generativeai as genai
import fitz  # PyMuPDF

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, List


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")


if not api_key:
    logging.error("FATAL: GEMINI_API_KEY not found in environment variables.")
else:
    genai.configure(api_key=api_key)


llm = genai.GenerativeModel("gemini-1.5-flash")



def read_pdf_file(file_path: str) -> str:
    """Reads text content from a PDF file."""
    try:
        doc = fitz.open(file_path)
        text = ""
        for page in doc:
            text += page.get_text()
        return text
    except Exception as e:
        logging.error(f"Error reading PDF file at {file_path}: {e}")
        return ""

def get_summary(text: str) -> str:
    """Generates a simple, readable summary of the provided text."""
    try:
        prompt = f"""
        You are an excellent paralegal in a big law firm.
        Summarize the following legal document in a simple, easily readable format.
        The summary should be clear enough for a person with little legal knowledge to understand.

        The legal document is:
        ---
        {text}
        ---
        """
        response = llm.generate_content(prompt)
        return response.text
    except Exception as e:
        logging.error(f"Error with Gemini API during summary generation: {e}")
        return "Could not generate summary due to an API error."

def extract_clauses(text: str) -> str:
    """
    Extracts key clauses and returns them as a clean JSON string.
    This function has a robust prompt to ensure valid JSON output.
    """
    try:
        prompt = f"""
        Analyze the following legal document. Your task is to extract key clauses and categorize them.

        **Instructions:**
        1.  Identify clauses related to "liability", "termination", and "confidentiality".
        2.  Format your entire output as a single, valid JSON object.
        3.  The keys of the object must be "liability", "termination", and "confidentiality".
        4.  The value for each key must be an array of strings, where each string is a direct quote of a clause from the document.
        5.  If no clauses are found for a category, the value must be an empty array [].
        6.  **CRITICAL: Do NOT include any text, explanations, or markdown formatting (like ```json) before or after the JSON object.** Your response must begin with `{{` and end with `}}`.

        Legal Document Text:
        ---
        {text}
        ---
        """
        response = llm.generate_content(prompt)
        
        # Clean up potential markdown formatting from the response text
        clean_text = response.text.strip().replace("```json", "").replace("```", "")
        
        return clean_text
        
    except Exception as e:
        logging.error(f"Error with Gemini API during clause extraction: {e}")
        # On error, return a valid, empty JSON string with an error message
        return json.dumps({
            "liability": ["An error occurred while trying to extract clauses from the document."],
            "termination": [],
            "confidentiality": []
        })


# --- 3. FASTAPI APPLICATION ---

app = FastAPI(title="Legal-Lens API")

# Configure CORS to allow requests from your React frontend
origins = [
    "http://localhost:5173",  # Default port for Vite React apps
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define the structure of the expected JSON response
class ResultStructure(BaseModel):
    summary: str
    clauses: Dict[str, List[str]]

@app.post("/simplify_document", response_model=ResultStructure)
async def simplify_document(uploaded_file: UploadFile = File(...)):
    """
    Main endpoint to receive a document, analyze it, and return results.
    """
    # Create a temporary path to save the uploaded file
    file_path = f"temp_{uploaded_file.filename}"
    
    try:
        # Save the uploaded file to the temporary path
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(uploaded_file.file, buffer)

        # Read the text from the saved file
        doc_text = read_pdf_file(file_path)

        # If reading fails, doc_text will be empty. Raise an error.
        if not doc_text:
            logging.error("Document text is empty. The file might be unreadable or empty.")
            raise HTTPException(status_code=400, detail="Could not read text from the uploaded document.")

        # Proceed with analysis
        summary = get_summary(doc_text)
        clauses_json_string = extract_clauses(doc_text)

        # Safely parse the JSON string for clauses
        try:
            clauses_dict = json.loads(clauses_json_string)
        except json.JSONDecodeError:
            logging.error("Failed to parse clauses JSON from AI response. The response was not valid JSON.")
            raise HTTPException(status_code=500, detail="The AI failed to structure the clauses correctly. Please try again.")

        return {"summary": summary, "clauses": clauses_dict}

    except HTTPException as http_exc:
        # Re-raise HTTPException so FastAPI handles it
        raise http_exc
    except Exception as e:
        # Catch any other unexpected errors
        logging.error(f"An unexpected server error occurred: {e}")
        raise HTTPException(status_code=500, detail="An internal server error occurred during analysis.")
    finally:
        # Always clean up and delete the temporary file
        if os.path.exists(file_path):
            os.remove(file_path)
