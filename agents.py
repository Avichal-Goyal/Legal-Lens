import os
from dotenv import load_dotenv
import google.generativeai as genai
import fitz
import logging

logging.basicConfig(level=logging.INFO)

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    logging.error("GEMINI_API_KEY not found in environment variables.")
else:
    genai.configure(api_key=api_key)


def read_pdf_file(file_path: str) -> str:
    try:
        doc = fitz.open(file_path)
        text = ""
        for page in doc:
            text += page.get_text()
        return text
    except Exception as e:
        logging.error(f"Error reading pdf file: {e}")
        return ""
llm = genai.GenerativeModel("gemini-1.5-flash")
def get_summary(text: str) -> str:
    try:
        prompt = f"You are an excelent paralegal in a big law firm.\nYou have to summarize the legal document in an easily readable format. You have to make the summary so that a person who doesnt know much about law can read it.\nThe legal document is:\n\n {text}"
        response = llm.generate_content(prompt)
        return response.text
    except Exception as e:
        logging.error(f"Error with Gemini API: {e}")
        return "Failed to get summary due to erro in fetching gemini api"

def extract_clauses(text: str) -> str:
    try:
        prompt = f"You are an excellent paralegal or legal assistant. It is difficult to read big legal documents.\nWe want to get all the legal clauses and rules mentioned in the document seperately. This will help in reading the document fast and easy.\nPresent all the legal clauses and rules mentioned in the following document properly:\n\n {text}"
        response = llm.generate_content(prompt)
        return response.text
    except Exception as e:
        logging.error(f"Error with Gemini API: {e}")
        return "Failed to get clauses due to erro in fetching gemini api"

