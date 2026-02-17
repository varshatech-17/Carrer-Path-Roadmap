import os
import json
import pdfplumber
from dotenv import load_dotenv
from openai import OpenAI
from groq import Groq

load_dotenv()

AI_PROVIDER = os.getenv("AI_PROVIDER", "groq").lower()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

def extract_text_from_pdf(file_file):
    """
    Extracts text from a PDF file object (UploadFile).
    """
    try:
        with pdfplumber.open(file_file) as pdf:
            text = ""
            for page in pdf.pages:
                text += page.extract_text() or ""
            return text
    except Exception as e:
        print(f"Error reading PDF: {e}")
        return ""

def get_ai_client():
    if AI_PROVIDER == "openai":
        return OpenAI(api_key=OPENAI_API_KEY), "gpt-4o"
    else:
        return Groq(api_key=GROQ_API_KEY), "llama-3.3-70b-versatile"

def analyze_career_gap(resume_text, dream_job):
    client, model = get_ai_client()
    
    system_prompt = """
    You are an expert Senior Technical Career Mentor and Agentic AI System. 
    Your goal is to perform a deep skill gap analysis for a student given their Resume and a Dream Job Title.
    
    You must output a strictly valid JSON object. Do not include markdown formatting like ```json ... ```. 
    Just output the raw JSON.
    
    The JSON structure must be:
    {
      "candidate_skills": ["List", "of", "skills", "found", "in", "resume"],
      "required_skills": ["List", "of", "top", "skills", "required", "for", "dream_job"],
      "missing_skills": ["List", "of", "critical", "skills", "missing"],
      "gap_analysis": "A brief, encouraging but realistic 2-3 sentence analysis of the gap.",
      "resources": [
        {
          "title": "Resource Name",
          "url": "Valid URL to a FREE resource (Youtube, Coursera Audit, Documentation, etc.)",
          "type": "Video/Course/Doc"
        }
        // Exactly 5 top quality FREE resources
      ],
      "roadmap": [
        {
          "week": "Week 1",
          "theme": "Theme of the week",
          "tasks": ["Task 1", "Task 2", "Task 3"]
        }
        // Generate a 4-8 week roadmap depending on the gap size
      ]
    }
    
    Focus on FREE resources only. Be specific.
    """
    
    user_prompt = f"""
    Resume Text:
    {resume_text[:4000]} 
    
    Dream Job: {dream_job}
    
    Analyze the gap and generate the JSON response.
    """
    
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.2,
            response_format={"type": "json_object"}
        )
        content = response.choices[0].message.content
        return json.loads(content)
    except Exception as e:
        print(f"AI Error: {e}")
        # Return dummy data if AI fails (for testing/demo without keys)
        return {
            "candidate_skills": ["Python (Detected)", "Communication"],
            "required_skills": ["Advanced AI", "System Design", "Cloud Ops"],
            "missing_skills": ["Vector DBs", "RAG Pipelines", "Kubernetes"],
            "gap_analysis": "Error connecting to AI (Check API Keys). Showing demo mode. You have a good start but need more production ML experience.",
            "resources": [
                {"title": "FastAPI Full Course", "url": "https://www.youtube.com/watch?v=0sOvCWFmrtA", "type": "Video"},
                {"title": "HuggingFace NLP Course", "url": "https://huggingface.co/learn/nlp-course", "type": "Course"},
                {"title": "Docker for Beginners", "url": "https://www.docker.com/101-tutorial", "type": "Doc"},
                {"title": "System Design Primer", "url": "https://github.com/donnemartin/system-design-primer", "type": "Repo"},
                {"title": "LangChain Docs", "url": "https://python.langchain.com/docs/get_started/introduction", "type": "Doc"}
            ],
            "roadmap": [
                 {"week": "Week 1", "theme": "Foundations", "tasks": ["Learn Docker", "Setup FastAPI"]},
                 {"week": "Week 2", "theme": "Advanced AI", "tasks": ["Build RAG pipeline", "Study Vector DBs"]}
            ]
        }
