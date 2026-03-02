import os
import whisper
from google import genai
from dotenv import load_dotenv

load_dotenv()

def transcribe_audio(file_path: str) -> str:
    model = whisper.load_model("base")
    result = model.transcribe(file_path)
    return result["text"]

def process_transcript(transcript: str) -> dict:
    client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
    model_id = "gemini-2.5-flash"
    
    intent_prompt = f"""You are a precise AI content architect. Analyze this transcript (can be Marathi, Hindi, English, or mixed).
CRITICAL RULES:
- Translate all concepts and extract the core thesis into strictly culturally neutral ENGLISH.
- Identify the exact INTENT (e.g., Technical, Satirical, Nonsense).
- Extract in ENGLISH: Core Thesis, Problem, Solution, and Key Technical Points.
- Eliminate filler words and ambiguity.
Transcript: {transcript}"""
    
    intent_res = client.models.generate_content(model=model_id, contents=intent_prompt).text
    
    outline_prompt = f"""Create a strict, logical content outline in ENGLISH based on the extracted analysis.
CRITICAL RULES:
- Reorganize non-linear thoughts into a clear "Speech-to-Structure" flow.
- Format with clear Markdown headings and sub-bullets.
- Ensure no ambiguity is passed forward.
Analysis: {intent_res}"""
    
    outline_res = client.models.generate_content(model=model_id, contents=outline_prompt).text
    
    draft_prompt = f"""Draft a publication-ready "Golden Source" article strictly in ENGLISH.
CRITICAL RULES:
- ZERO HALLUCINATION: Do not invent profound psychological or academic themes (like "The Performance Paradox") 
- The output MUST be in culturally neutral ENGLISH, regardless of the source transcript language.
- Tone: Professional, authoritative, and brand-consistent.
- Eliminate all spoken-language artifacts.
- The text must be structurally sound and unambiguous for downstream multilingual CMS translation.
Outline: {outline_res}
Source Transcript: {transcript}"""
    
    draft_res = client.models.generate_content(model=model_id, contents=draft_prompt).text
    
    return {"intent": intent_res, "outline": outline_res, "draft": draft_res}
