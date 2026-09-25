import os
import json
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from google import genai

BASE_DIR = Path(__file__).resolve().parent

load_dotenv(BASE_DIR / ".env")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite").strip()

app = FastAPI(title="EduGenie API", version="1.0.0")

# Needed when EduGenie.html is opened directly with file://
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

client = genai.Client(api_key=GEMINI_API_KEY) if GEMINI_API_KEY else None


def clean_topic(value):
    return "" if value is None else str(value).strip()


def ai_text(prompt: str) -> str:
    if not GEMINI_API_KEY:
        raise RuntimeError(
            "Gemini API key is not configured. Add GEMINI_API_KEY to .env."
        )

    if client is None:
        raise RuntimeError("Gemini client could not be initialized.")

    try:
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
        )
        text = getattr(response, "text", None)

        if not text:
            raise RuntimeError("Gemini returned an empty response.")

        return text.strip()

    except Exception as exc:
        print("Gemini API error:", exc)
        raise RuntimeError(f"Gemini API error: {exc}")


@app.get("/")
async def home():
    html_file = BASE_DIR / "EduGenie.html"

    if not html_file.exists():
        return JSONResponse(
            {"error": "EduGenie.html not found."},
            status_code=404,
        )

    return FileResponse(html_file, media_type="text/html")


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "gemini_configured": bool(GEMINI_API_KEY),
        "model": GEMINI_MODEL,
    }


@app.post("/qa")
async def ask_ai(payload: dict):
    question = clean_topic(
        payload.get("question")
        or payload.get("prompt")
        or payload.get("topic")
    )

    if not question:
        return JSONResponse(
            {"error": "Please enter a question."},
            status_code=400,
        )

    prompt = f"""
You are EduGenie, an AI learning assistant.

Answer the student's question clearly and accurately.

Student question:
{question}

Requirements:
- Explain in simple language.
- Use examples when useful.
- Keep the answer student-friendly.
- Use headings and bullet points where appropriate.
"""

    try:
        return {"answer": ai_text(prompt)}
    except RuntimeError as exc:
        return JSONResponse({"error": str(exc)}, status_code=503)


@app.post("/explain")
async def explain(payload: dict):
    topic = clean_topic(payload.get("topic"))
    level = clean_topic(payload.get("level") or "Beginner")

    if not topic:
        return JSONResponse(
            {"error": "Please enter a topic."},
            status_code=400,
        )

    prompt = f"""
Explain the following topic to a college student.

Topic:
{topic}

Learning level:
{level}

Structure:
1. Simple definition
2. How it works
3. Important concepts
4. Simple example
5. Real-world application
6. Short summary

Use clear and student-friendly language.
"""

    try:
        return {"answer": ai_text(prompt)}
    except RuntimeError as exc:
        return JSONResponse({"error": str(exc)}, status_code=503)


@app.post("/summarize")
async def summarize(payload: dict):
    content = clean_topic(payload.get("text"))

    if not content:
        return JSONResponse(
            {"error": "Please paste some text."},
            status_code=400,
        )

    prompt = f"""
Summarize the text below for a student.

Return:
- A 2-3 sentence summary
- 5 key points
- 5 important keywords

Text:
{content}
"""

    try:
        return {"answer": ai_text(prompt)}
    except RuntimeError as exc:
        return JSONResponse({"error": str(exc)}, status_code=503)


@app.post("/quiz")
async def quiz(payload: dict):
    topic = clean_topic(payload.get("topic"))

    if not topic:
        return JSONResponse(
            {"error": "Please enter a topic."},
            status_code=400,
        )

    prompt = f"""
Create exactly 3 multiple-choice questions about:
"{topic}"

Each question must have exactly 4 options and exactly one correct answer.

Return ONLY valid JSON in this format:

{{
  "topic": "{topic}",
  "questions": [
    {{
      "question": "...",
      "options": ["...", "...", "...", "..."],
      "answer": 0,
      "explanation": "..."
    }}
  ]
}}

The answer value must be 0, 1, 2, or 3.
Do not include markdown or ```json.
"""

    try:
        raw = ai_text(prompt)
        start = raw.find("{")
        end = raw.rfind("}")

        if start < 0 or end <= start:
            raise ValueError("No JSON object returned.")

        data = json.loads(raw[start:end + 1])

        if not isinstance(data.get("questions"), list):
            raise ValueError("Invalid quiz format.")

        return data

    except RuntimeError as exc:
        return JSONResponse({"error": str(exc)}, status_code=503)

    except Exception as exc:
        print("Quiz parsing error:", exc)
        return JSONResponse(
            {"error": "The AI returned an invalid quiz. Please try again."},
            status_code=502,
        )


@app.post("/learn/recommendations")
async def recommendations(payload: dict):
    topic = clean_topic(payload.get("topic"))
    goal = clean_topic(
        payload.get("goal") or "college-level understanding"
    )

    if not topic:
        return JSONResponse(
            {"error": "Please enter a subject or skill."},
            status_code=400,
        )

    prompt = f"""
Create a practical learning roadmap for:

"{topic}"

Learner goal:
{goal}

Use exactly three stages:

BEGINNER
INTERMEDIATE
ADVANCED

For each stage include:
- Concepts
- Practice task
- Mini project
- Checkpoint

Finish with 5 useful resource types to look for.

Keep it realistic for a student.
"""

    try:
        return {"answer": ai_text(prompt)}
    except RuntimeError as exc:
        return JSONResponse({"error": str(exc)}, status_code=503)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
    )
