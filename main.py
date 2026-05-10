from fastapi import FastAPI, UploadFile, File
import shutil
import os

from pypdf import PdfReader
import docx

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = FastAPI()

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ---------------- HOME ----------------
@app.get("/")
def home():
    return {"message": "Backend is working"}

# ---------------- FILE UPLOAD ----------------
@app.post("/upload/")
def upload_file(file: UploadFile = File(...)):

    file_path = os.path.join(UPLOAD_FOLDER, file.filename)

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    return {
        "filename": file.filename,
        "status": "uploaded successfully"
    }

# ---------------- TEXT EXTRACTION ----------------
@app.get("/extract/")
def extract_text(filename: str):

    file_path = os.path.join(UPLOAD_FOLDER, filename)

    if not os.path.exists(file_path):
        return {"error": "File not found"}

    text = ""

    # PDF
    if filename.endswith(".pdf"):
        reader = PdfReader(file_path)

        for page in reader.pages:
            text += page.extract_text() or ""

    # DOCX
    elif filename.endswith(".docx"):
        doc = docx.Document(file_path)

        for para in doc.paragraphs:
            text += para.text + "\n"

    # TXT
    elif filename.endswith(".txt"):

        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()

    else:
        return {"error": "Unsupported file format"}

    return {
        "filename": filename,
        "extracted_text": text[:1000]
    }

# ---------------- PLAGIARISM CHECK ----------------
@app.post("/plagiarism/")
def check_plagiarism(text1: str, text2: str):

    documents = [text1, text2]

    vectorizer = TfidfVectorizer()

    vectors = vectorizer.fit_transform(documents)

    similarity = cosine_similarity(vectors[0], vectors[1])[0][0]

    percentage = round(similarity * 100, 2)

    return {
        "similarity_percentage": percentage
    }