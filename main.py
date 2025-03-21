
import re
import os
from pydantic import BaseModel
import json
from fastapi import FastAPI, HTTPException
from langchain_groq import ChatGroq
from langchain.agents import initialize_agent, AgentType
from langchain.memory import ConversationBufferMemory
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from video_summary_service import (
    get_youtube_video,
    get_youtube_transcript,
    generate_summary,
    analyze_sentiment_service,
    extract_character_mentions,
    generate_audio_summary,
    extract_summary_data,
)
# ✅ FastAPI Endpoint
app = FastAPI()
# Enable CORS for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
if not os.path.exists("static"):
    os.makedirs("static")

# Mount the static directory to serve audio files
app.mount("/static", StaticFiles(directory="static"), name="static")

# ✅ Define AI model using Groq
llm = ChatGroq(
    model="llama3-8b-8192",
    temperature=0.1,
    max_tokens=None,
    max_retries=2
)

# ✅ Define tools list
tools = [
    get_youtube_video,
    get_youtube_transcript,
    generate_summary,
    analyze_sentiment_service,
    extract_character_mentions,
    generate_audio_summary,
]

# ✅ Initialize LangChain Agent with tools
llm_agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    memory=ConversationBufferMemory(memory_key="chat_history"),
    max_iterations=30,
)


# ✅ Define input schema
class VideoRequest(BaseModel):
    video_title: str


@app.post("/summarize")
async def summarize_video(request: VideoRequest):
    """Summarizes a YouTube video given its title."""
    video_title = request.video_title  # Extract from validated input
    prompt = f"""
        You are an AI assistant that summarizes YouTube videos.
        Given the video titled: "{video_title}", provide a structured summary strictly in JSON format with these keys:
        - "core_message": (Main idea of the video)
        - "storyline": (Plot or storyline, if applicable)
        - "key_takeaways": (List of main lessons or takeaways)
        - "sentiments": (Emotional tone of the video)
        - "notable_characters": (List of notable characters in the video)
        - "audio_file": (Filename of the generated audio summary)
        - "video_link": (Link to the video)
        """
    try:
        response = llm_agent.invoke(prompt)
        # ✅ Check if the response is already a dictionary
        if isinstance(response, dict):
            parsed_response = response  # Use as-is
        elif isinstance(response, str):
            raw_text = response.strip()  # Remove any surrounding spaces/newlines

            # 🔥 Extract JSON content using regex (Fix for LangChain output formatting issues)
            match = re.search(r"\{.*\}", raw_text, re.DOTALL)

            if not match:
                raise HTTPException(status_code=500, detail="Failed to extract valid JSON from AI response.")

            cleaned_json_str = match.group(0)  # Extract JSON part only
            parsed_response = json.loads(cleaned_json_str)  # Convert to Python dictionary
        else:
            raise HTTPException(status_code=500, detail="Unexpected response type from LLM.")

        # ✅ Generate audio for the extracted summary
        print(parsed_response)
        summary = extract_summary_data(parsed_response)
        print(summary)
        #audio_file = generate_audio_summary(parsed_response.get("core_message", ""))
        #parsed_response["audio_file"] = audio_file  # Attach audio file to response

        return summary

    except json.JSONDecodeError as e:
        raise HTTPException(status_code=500, detail=f"Failed to parse AI response: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

# ✅ Run server if executed directly
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
