from datetime import datetime
import threading
from pydantic import BaseModel
import os
import json
import logging
from concurrent.futures import ThreadPoolExecutor
from youtube_transcript_api import YouTubeTranscriptApi
from gtts import gTTS
from dotenv import load_dotenv
from groq import Groq
from serpapi import GoogleSearch
import serpapi
from textblob import TextBlob
import re

load_dotenv()

# Load API Keys
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
SERPAPI_KEY = os.getenv("SERPAPI_API_KEY")

app_domain = os.getenv("APP_DOMAIN")

# Initialize Groq client
groq_client = Groq(api_key=GROQ_API_KEY)
# client = serpapi.Client(api_key=os.getenv("SERPAPI_API_KEY"))

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Define AI function-calling functions
functions = [
    {
        "name": "search_youtube_video",
        "description": "Search for a YouTube video using SerpAPI based on a given title or keyword.",
        "parameters": {
            "type": "object",
            "properties": {
                "title": {
                    "type": "string",
                    "description": "The title or keywords of the YouTube video",
                }
            },
            "required": ["title"],
        },
    },
    {
        "name": "get_youtube_transcript",
        "description": "Retrieve the transcript of a YouTube video given its video ID.",
        "parameters": {
            "type": "object",
            "properties": {
                "video_id": {"type": "string", "description": "The YouTube video ID"}
            },
            "required": ["video_id"],
        },
    },
    {
        "name": "summarize_transcript",
        "description": "Summarize a given transcript using Llama 3.3 and extract key plot twists.",
        "parameters": {
            "type": "object",
            "properties": {
                "transcript": {
                    "type": "string",
                    "description": "The full transcript of the video",
                }
            },
            "required": ["transcript"],
        },
    },
]

# Function Implementations


def extract_videoid(url: str) -> str:
    """Extract YouTube video ID from URL"""
    regex = r"(?:v=|/)([0-9A-Za-z-]{11}).*"
    match = re.search(regex, url)
    return match.group(1) if match else None


def search_youtube_video(title: str):
    logger.info(f"Searching YouTube for: {title}")
    search = GoogleSearch(
        {"engine": "youtube", "search_query": title, "api_key": SERPAPI_KEY}
    )
    response = search.get_dict()

    if "video_results" in response and response["video_results"]:
        video = response["video_results"][0]
        video_id = extract_videoid(video["link"])
        logger.info(
            f"Found video: {video['title']}, url: {video['link']}, ID: {video_id}"
        )
        return {
            "video_id": video_id,
            "video_url": video["link"],
            "title": video["title"],
            "channel": (
                video.get("channel").get("name") if video.get("channel") else None
            ),
        }
    return None


def get_youtube_transcript(video_id: str):
    logger.info(f"Fetching transcript for video ID: {video_id}")
    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        full_text = " ".join([t["text"] for t in transcript])
        print(f"transcript found: \n {full_text[:60]}")
        return full_text[:6000]
    except:
        return "Transcript not available."


def extract_summary_and_twists(content: str):
    """
    Parses the AI response to extract summary and key plot twists from the content.
    """
    summary = ""
    plot_twists = []

    if "**Summary:**" in content:
        parts = content.split("**Summary:**")
        summary = (
            parts[1].split("**Key Plot Twists:**")[0].strip()
            if "**Key Plot Twists:**" in parts[1]
            else parts[1].strip()
        )

    if "**Key Plot Twists:**" in content:
        twists_part = content.split("**Key Plot Twists:**")[1].strip()
        plot_twists = [
            line.strip()
            for line in twists_part.split("\n")
            if line.strip() and not line.startswith("*")
        ]

    return summary, plot_twists


def summarize_transcript(transcript: str):
    if transcript == "Transcript not available.":
        return "Sorry, no transcript is available for this video."

    logger.info("Summarizing transcript and extracting plot twists...")
    response = groq_client.chat.completions.create(
        model="llama3-70b-8192",
        messages=[
            {
                "role": "system",
                "content": "You are an AI that processes YouTube videos into summaries and extracts key plot twists.",
            },
            {
                "role": "user",
                "content": f"Summarize the following text and identify key plot twists: {transcript}",
            },
        ],
        # functions=functions,
    )
    summary_message = response.choices[0].message
    content = summary_message.content or ""
    summary, plot_twists = extract_summary_and_twists(content)
    logger.info("Extracted summary directly from content.")
    return summary, plot_twists, content


def analyze_sentiment(text: str):
    sentiment = TextBlob(text).sentiment.polarity
    print("\n sentiment:", sentiment, TextBlob(text).sentiment)
    if sentiment > 0:
        return "Positive"
    elif sentiment < 0:
        return "Negative"
    else:
        return "Neutral"


def extract_character_mentions(text: str):
    names = set(re.findall(r"\b[A-Z][a-z]+\b", text))
    return list(names)


def generate_audio_summary(summary: str, filename: str):
    logger.info("Generating audio summary...")

    # Ensure the 'static' directory exists
    os.makedirs("static", exist_ok=True)

    tts = gTTS(summary, lang="en")
    audio_path = f"static/{filename}.mp3"
    tts.save(audio_path)
    return filename


# Function to trigger TTS in a separate thread
def async_generate_audio(summary: str, filename: str):
    threading.Thread(
        target=generate_audio_summary, args=(summary, filename), daemon=True
    ).start()


def execute_function_call(function_name, arguments, messages: list[dict[str, str]]):
    if function_name == "search_youtube_video":
        search_result = search_youtube_video(**arguments)
        print("Search results", search_result)
        messages.append(
            {
                "role": "assistant",
                "content": f"Found video: {json.dumps(search_result)}",
            }
        )
        messages.append(
            {
                "role": "user",
                "content": f"Extract transcript for video ID: {search_result['video_id']}",
            }
        )
        return search_result
    elif function_name == "get_youtube_transcript":
        return get_youtube_transcript(**arguments)
    elif function_name == "summarize_transcript":
        summarize_resp = summarize_transcript(**arguments)
        return summarize_resp
    return None


def process_results_response(results: dict, tts: bool):
    summary, plot_twists, content = results.get("summarize_transcript", ("", "", ""))
    sentiment = analyze_sentiment(summary)
    characters = extract_character_mentions(summary)
    # Generate unique filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    video_id = results.get("search_youtube_video", {}).get("video_id", "")
    filename = f"{video_id}_{timestamp}"
    if tts:
        # generate_audio_summary(summary, filename)
        async_generate_audio(summary, filename)

    search_video_results = results.get("search_youtube_video", {})

    return {
        "title": search_video_results["title"],
        "link": search_video_results["video_url"],
        "video_url": search_video_results["video_url"],
        "channel": search_video_results["channel"],
        "summary_with_twists": content,
        "summary": summary,
        "sentiment": sentiment,
        "characters": characters,
        "plot_twists": plot_twists,
        "audio_filename": f"{filename}.mp3" if tts else None,
        "audio_summary_url": (
            f"{app_domain}/download-audio/{filename}.mp3" if tts else None
        ),
    }


def analyze_video_with_function_calls(query: str, tts: bool = False) -> dict:
    messages = [{"role": "user", "content": query}]
    results = {}
    count = 0
    max_attempts = 3  # Prevent infinite loops

    with ThreadPoolExecutor() as executor:
        while count < max_attempts:
            response = groq_client.chat.completions.create(
                model="llama3-70b-8192",
                messages=messages,
                functions=functions,
                # max_tokens=4096,
            )

            choice = response.choices[0].message
            print(f"\n {count} response...")
            tool_calls = choice.tool_calls
            if tool_calls:
                tool_call = tool_calls[0]
                print(f"{count} response tool call...")
                function_name = tool_call.function.name
                arguments = json.loads(tool_call.function.arguments)

                # Execute function in parallel
                future = executor.submit(
                    execute_function_call, function_name, arguments, messages
                )
                result = future.result()
                # result = execute_function_call(function_name, arguments, messages)
                results[function_name] = result

                # Provide the function result to the model
                messages.append(
                    {
                        "role": "assistant",
                        "content": choice.content,
                        "function_call": tool_call,
                    }
                )
                messages.append(
                    {
                        "role": "function",
                        "name": function_name,
                        "content": json.dumps(result),
                    }
                )
                count += 1
            else:
                return process_results_response(results, tts)

    return process_results_response(results, tts)
