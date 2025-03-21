from email.mime import audio
import json
import re
import requests
import uuid
from langchain.tools import tool
from pydantic import BaseModel
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from googleapiclient.discovery import build
from youtube_transcript_api import YouTubeTranscriptApi
from textblob import TextBlob
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from deepgram import DeepgramClient, SpeakOptions
import os


# ✅ Load environment variables (API keys)
load_dotenv()
YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")
DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")

# ✅ Initialize AI model (Groq Llama3)
llm = ChatGroq(
    model="llama3-8b-8192",
    temperature=0.1,  # More deterministic output
    max_tokens=None,
    max_retries=2
)

# Initialize YouTube API client
youtube = build("youtube", "v3", developerKey=YOUTUBE_API_KEY)

class VideoRequest(BaseModel):
    title: str


@tool
def extract_character_mentions(text: str):
    """Finds and extracts notable characters from the summary using regex."""
    print("extract_character_mentions");
    # Look for capitalized words (which may be names)
    matches = re.findall(r"(?<!\.)\b[A-Z][a-z]+(?:\s[A-Z][a-z]+)*\b", text)
    
    # Return unique names
    return list(set(matches))
@tool
def analyze_sentiment_service(text: str):
    """Analyzes the sentiment of a given text (Positive, Negative, Neutral)."""
    print("analyze_sentiment_service")
    sentiment_score = TextBlob(text).sentiment.polarity
    if sentiment_score > 0:
        return "Positive"
    elif sentiment_score < 0:
        return "Negative"
    return "Neutral"

@tool
def get_youtube_transcript(video_id: str):
    """Fetches the transcript of a YouTube video given its ID."""
    print("get_youtube_transcript")
    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        transcript_text = " ".join([entry["text"] for entry in transcript])
        
        # Return only first 6000 characters (to fit model input limits)
        return transcript_text[:6000]
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching transcript: {str(e)}")

@tool
def generate_audio_summary(summary_text: str):
    """Converts text to speech using Deepgram API."""
    
    print("generate_audio_summary")
    if not summary_text.strip():  # Check if text is empty
        raise ValueError("Summary text is empty, cannot generate audio.")
    unique_id = uuid.uuid4().hex  # Generate a unique identifier
    filename = f"static/summary_audio_{unique_id}.mp3"
    
    try:
        # Initialize Deepgram client
        deepgram = DeepgramClient(api_key=DEEPGRAM_API_KEY)

        # Configure TTS options
        options = SpeakOptions(
            model="aura-asteria-en",  # Natural-sounding English model
        )

        # Generate and save the audio file
        response = deepgram.speak.rest.v("1").save(filename, {"text": summary_text}, options)
        
        print(response.to_json(indent=4))  # Debugging output
        
        # Return the filename or file path
        return filename
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Deepgram TTS failed: {str(e)}")
@tool
def get_youtube_video(title: str):
    """Fetches the most relevant YouTube video based on the title."""
    print("get_youtube_video")
    try:
        request = youtube.search().list(
            q=title,
            part="snippet",
            maxResults=1,
            type="video"
        )
        response = request.execute()
        
        if not response["items"]:
            return None
        
        video = response["items"][0]
        video_id = video["id"]["videoId"]
        video_title = video["snippet"]["title"]
        channel = video["snippet"]["channelTitle"]
        link = f"https://www.youtube.com/watch?v={video_id}"

        return {"title": video_title, "channel": channel, "link": link}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching video: {str(e)}")
@tool
def generate_summary(video_title: str):
    """Generates a structured summary using LangChain."""
    response_schemas = [
        ResponseSchema(name="core_message", description="Main idea of the video"),
        ResponseSchema(name="storyline", description="Plot or storyline (if applicable)"),
        ResponseSchema(name="key_takeaways", description="Main lessons or takeaways"),
        ResponseSchema(name="sentiments", description="Emotional tone of the video"),
        ResponseSchema(name="notable_characters", description="Key characters in the video")
    ]
    
    output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
    prompt_template = ChatPromptTemplate.from_template(
        """
        You are an AI assistant that summarizes YouTube videos.
        Given the video titled: "{video_title}", provide a structured summary including:
        - Core Message
        - Storyline (if applicable)
        - Key Takeaways
        - Sentiments
        - Notable Characters
        """
    )
    
    prompt = prompt_template.format(video_title=video_title)
    response = llm.invoke(prompt)
    raw_response = response.content.strip()
    audio_file = generate_audio_summary(raw_response)
    
    if not raw_response:
        return {"error": "LLM returned an empty response"}

    # Extract relevant parts using regex
    match = re.search(r"\*\*Core Message:\*\*\s*(.+?)\n", raw_response)
    core_message = match.group(1) if match else "Not found"

    match = re.search(r"\*\*Storyline:\*\*\s*(.+?)\n\n", raw_response, re.DOTALL)
    storyline = match.group(1) if match else "Not found"

    key_takeaways = re.findall(r"\d+\.\s*(.+)", raw_response)

    match = re.search(r"\*\*Sentiments:\*\*\s*(.+)", raw_response, re.DOTALL)
    sentiments = match.group(1) if match else "Not found"
    
    match = re.search(r"\*\*Notable Characters:\*\*\s*(.+)", raw_response, re.DOTALL)
    notable_characters = match.group(1) if match else "Not found"

    # Return as a proper JSON response
    summary_data = {
        "title": video_title,
        "core_message": core_message,
        "storyline": storyline,
        "key_takeaways": key_takeaways,
        "sentiments": sentiments,
        "notable_characters": notable_characters,
        "audio_file": audio_file
    }

    return summary_data

def extract_summary_data(response_json):
    """
    Cleans the response JSON by removing extra text before retrieving the 'output' field,
    then parsing it into a valid JSON object.

    Steps:
    1. Extracts the 'output' field from the response.
    2. Removes any text before/after the actual JSON using regex.
    3. Fixes escape characters to ensure proper JSON formatting.
    4. Parses the cleaned JSON string into a dictionary.

    Args:
        response_json (dict): The original JSON response containing an 'output' field.

    Returns:
        dict: A properly parsed JSON object with the summary details or None if parsing fails.
    """
    try:
        # Extract the 'output' field (if it exists)
        output_text = response_json.get("output", "")

        if not output_text:
            raise ValueError("No 'output' field found in response.")

        # Use regex to extract ONLY the JSON object (handles extra text around it)
        json_match = re.search(r'\{.*\}', output_text, re.DOTALL)

        if not json_match:
            raise ValueError("No valid JSON found in 'output' field.")

        # Extract only the JSON part
        json_string = json_match.group(0).strip()

        # Ensure proper formatting by replacing incorrectly escaped quotes
        json_string = json_string.replace('\\"', '"')

        # Fix cases where newlines or unnecessary characters break JSON formatting
        json_string = re.sub(r'\\n', '', json_string)  # Remove stray `\n` characters

        # Parse the extracted JSON string into a dictionary
        parsed_output = json.loads(json_string)

        return parsed_output  # Return the cleaned JSON object

    except (json.JSONDecodeError, ValueError) as e:
        print(f"Error parsing JSON: {e}")
        return None  # Return None if parsing fails


