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
import spacy

# Load a small NLP model for Named Entity Recognition (NER)
nlp = spacy.load("en_core_web_sm")
# ✅ Load environment variables (API keys)
load_dotenv()
YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY_2")
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
    """Extracts notable characters from the summary using regex and NLP."""
    print("extract_character_mentions")

    # Regex to capture proper names (multi-word names allowed)
    regex_pattern = r"\b(?:[A-Z][a-z]+(?:\s[A-Z][a-z]+)*)\b"
    matches = re.findall(regex_pattern, text)

    # Filter out common words that start with capital letters (e.g., "The", "It")
    common_words = {"The", "A", "It", "And", "He", "She", "They", "In"}
    filtered_matches = [name for name in matches if name not in common_words]

    # Apply Named Entity Recognition (NER) for backup extraction
    doc = nlp(text)
    ner_names = [ent.text for ent in doc.ents if ent.label_ in ["PERSON", "ORG"]]

    # Combine regex and NLP results, removing duplicates
    character_names = list(set(filtered_matches + ner_names))

    return character_names  # Return a unique list of character names

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
def get_youtube_transcript(video_url: str):
    """Fetches the transcript of a YouTube video given its ID."""
    print("get_youtube_transcript")
     # Extract video ID from the URL using regex
    match = re.search(r"v=([a-zA-Z0-9_-]+)", video_url)
    if match:
        video_id = match.group(1)
    else:
        video_id = video_url  # Assume it's already a video ID

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

import json
import re

def extract_summary_data(response_json):
    """
    Cleans the response JSON by extracting and parsing the 'output' field.

    Steps:
    1. Extracts the 'output' field from the response.
    2. If "AI:" is present, extracts the content after it.
    3. Uses regex to isolate only the JSON object.
    4. Fixes escape characters and ensures proper formatting.
    5. Parses the cleaned JSON string into a dictionary.

    Args:
        response_json (dict): The original JSON response containing an 'output' field.

    Returns:
        dict: A properly parsed JSON object with the summary details or None if parsing fails.
    """
    try:
        # Step 1: Extract the 'output' field
        output_text = response_json.get("output", "")

        if not output_text:
            raise ValueError("No 'output' field found in response.")

        # Step 2: Find "AI:" and extract text after it (if present)
        ai_match = re.search(r'AI:\s*(\{.*\})', output_text, re.DOTALL)
        if ai_match:
            output_text = ai_match.group(1)  # Extract everything after "AI:"

        # Step 3: Use regex to extract only the JSON object
        json_match = re.search(r'\{.*\}', output_text, re.DOTALL)
        if not json_match:
            raise ValueError("No valid JSON found in 'output' field.")

        # Step 4: Extract and clean the JSON string
        json_string = json_match.group(0).strip()

        # Fix incorrectly escaped quotes
        json_string = json_string.replace('\\"', '"')

        # Remove stray newlines
        json_string = re.sub(r'\\n', '', json_string)

        # Step 5: Parse the extracted JSON string into a dictionary
        parsed_output = json.loads(json_string)

        return parsed_output  # Return cleaned JSON object

    except (json.JSONDecodeError, ValueError) as e:
        print(f"Error parsing JSON: {e}")
        return None  # Return None if parsing fails




