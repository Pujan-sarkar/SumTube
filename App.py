from fastapi import FastAPI
from typing import Optional
import os
from dotenv import load_dotenv
from google import genai
from google.genai import types
from youtube_transcript_api import YouTubeTranscriptApi

# Load environment variables from .env file
load_dotenv()

app = FastAPI()

# --- Utility Functions ---

def extract_youtube_id(url: str) -> str:
    """
    Extracts the YouTube video ID from common YouTube URL formats.
    """
    if "youtube.com/watch?v=" in url:
        return url.split("v=")[1].split("&")[0]
    elif "youtu.be/" in url:
        return url.rstrip('/').split("/")[-1]
    else:
        raise ValueError("Invalid YouTube URL")

def fetch_youtube_transcript(video_id: str) -> Optional[str]:
    """
    Fetches and concatenates YouTube transcript. Returns None if not available.
    """
    try:
        ytt_api = YouTubeTranscriptApi()
        transcript = ytt_api.fetch(video_id).to_raw_data()
        all_text = " ".join(entry["text"] for entry in transcript)
        return all_text if all_text.strip() else None
    except Exception:
        return None

def summarize_transcript_with_gemini(transcript: str) -> str:
    """
    Summarizes the transcript using Gemini and returns JSON-formatted summary.
    """
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise EnvironmentError("GEMINI_API_KEY environment variable not set. Please set it in your .env file or system environment.")
    client = genai.Client(api_key=api_key)
    model = "gemini-2.5-pro"

    prompt = f"""Based on the following transcription please give me the summary in a json format like this:
{{
   "Topic name" : "name of the topic",
   "Topic summary" : "Summary of the topic"
}}
Transcription:
{transcript}
"""
    contents = [
        types.Content(
            role="user",
            parts=[types.Part.from_text(text=prompt)],
        )
    ]
    response = client.models.generate_content(model=model, contents=contents)
    return response.candidates[0].content.parts[0].text

# --- FastAPI Endpoint ---

@app.get("/summarize")
def get_summary(url: str):
    try:
        video_id = extract_youtube_id(url)
    except ValueError:
        return {"error": "Invalid YouTube URL."}

    transcript = fetch_youtube_transcript(video_id)
    if transcript:
        summary = summarize_transcript_with_gemini(transcript)
        return {"summary": summary}
    else:
        return {"error": "Transcript not found or not available for this video."}

# --- Run server for local testing ---

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
