import os
import fitz  # PyMuPDF
import sqlite3
from fastapi import FastAPI, UploadFile, File
from google import genai
from google.cloud import vision
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

app = FastAPI()

# --- CONFIG ---
DB_NAME = "evaluation_system.db"

# 🔐 API Keys (better: env variable use karo)
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "YOUR_API_KEY")

# Gemini Client
client = genai.Client(api_key=GEMINI_API_KEY)

# Google Vision
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "service-account-file.json"
vision_client = vision.ImageAnnotatorClient()

# Load BERT model
print("Loading BERT model...")
bert_model = SentenceTransformer('all-MiniLM-L6-v2')

# --- DATABASE ---
def init_db():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS questions (
            q_id TEXT PRIMARY KEY,
            question_text TEXT,
            max_marks REAL
        )
    ''')

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            student_name TEXT,
            q_id TEXT,
            obtained_marks REAL,
            ai_solution TEXT,
            similarity TEXT
        )
    ''')

    conn.commit()
    conn.close()

init_db()

# --- HELPER ---
def get_ai_solution(question):
    prompt = f"Provide a concise academic ideal answer for: {question}"

    response = client.models.generate_content(
        model="gemini-1.5-flash",
        contents=prompt
    )

    return response.text if response else "No response"

# --- ROUTES ---

@app.get("/")
def home():
    return {"message": "AI Evaluation System Running 🚀"}

@app.post("/add-questions")
def add_questions(q_id: str, text: str, marks: float):
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    cursor.execute(
        "INSERT OR REPLACE INTO questions VALUES (?, ?, ?)",
        (q_id, text, marks)
    )

    conn.commit()
    conn.close()

    return {"status": "Question Added"}

@app.post("/evaluate-pdf")
async def evaluate_pdf(student_name: str, file: UploadFile = File(...)):

    # Save temp file
    temp_path = f"temp_{file.filename}"
    with open(temp_path, "wb") as f:
        f.write(await file.read())

    # Extract text using OCR
    doc = fitz.open(temp_path)
    student_text = ""

    for page in doc:
        pix = page.get_pixmap()
        image = vision.Image(content=pix.tobytes("png"))

        res = vision_client.document_text_detection(image=image)

        if res.full_text_annotation:
            student_text += res.full_text_annotation.text

    doc.close()
    os.remove(temp_path)

    # Get questions
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM questions")
    questions = cursor.fetchall()

    report = []

    for q_id, q_text, max_marks in questions:

        # Gemini answer
        ideal_answer = get_ai_solution(q_text)

        # Similarity
        emb1, emb2 = bert_model.encode([student_text, ideal_answer])
        similarity = float(cosine_similarity([emb1], [emb2])[0][0])

        marks = round(similarity * max_marks, 2) if similarity > 0.3 else 0

        # Save result
        cursor.execute('''
            INSERT INTO results (student_name, q_id, obtained_marks, ai_solution, similarity)
            VALUES (?, ?, ?, ?, ?)
        ''', (
            student_name,
            q_id,
            marks,
            ideal_answer,
            f"{round(similarity * 100, 2)}%"
        ))

        report.append({
            "question_id": q_id,
            "marks": marks,
            "similarity": f"{round(similarity * 100, 2)}%"
        })

    conn.commit()
    conn.close()

    return {
        "student": student_name,
        "report": report
    }

@app.get("/view-results")
def view_results():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    cursor.execute("SELECT * FROM results")
    data = cursor.fetchall()

    conn.close()

    return {"results": data}