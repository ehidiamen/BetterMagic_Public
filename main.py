
from cProfile import run
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
from langchain.tools import Tool

from video_summary_service import (
    get_youtube_video,
    #get_youtube_transcript_data,
    generate_summary,
    analyze_sentiment_service,
    # extract_character_mentions,
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
#tools = [
#    get_youtube_video,
#    get_youtube_transcript,
#    generate_summary,
#    analyze_sentiment_service,
#    # extract_character_mentions,
#    generate_audio_summary,
#]

tools = [
    Tool(
        name="get_youtube_video",
        func=get_youtube_video,
        description="Search for a YouTube video based on a given title."
    ),
    #Tool(
    #    name="get_youtube_transcript",
    #    func=get_youtube_transcript_data,
    #    description="Retrieve the transcript of a given YouTube video."
    #),
    Tool(
        name="generate_summary",
        func=generate_summary,
        description="Summarize a given transcript into key points."
    ),
    Tool(
        name="analyze_sentiment_service",
        func=analyze_sentiment_service,
        description="Analyze the sentiment of the summarized text."
    ),
    Tool(
        name="generate_audio_summary",
        func=generate_audio_summary,
        description="Generate an audio file summarizing the video."
    ),
]

# ✅ Initialize LangChain Agent with tools
llm_agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    memory=ConversationBufferMemory(memory_key="chat_history"),
    max_iterations=15,
)

def execute_function_call(function_name, arguments, messages: list[dict[str, str]]):
    """Executes the correct tool function based on function_name."""

    # Handle YouTube search
    if function_name == "get_youtube_video":
        video_data = get_youtube_video(**arguments)
        
        # Append the response to the conversation
        messages.append({"role": "assistant", "content": f"Found video: {video_data['title']}, {video_data['link']}"})
        
        # Ensure the next step is only appended **once**
        if not any("Extract transcript for video ID" in msg["content"] for msg in messages):
            messages.append({"role": "user", "content": f"Extract transcript for video ID: {video_data['link'].split('=')[-1]}"})

        return video_data

    # Handle transcript retrieval
    #elif function_name == "get_youtube_transcript":
    #    return get_youtube_transcript_data(**arguments)

    # Handle summarization
    # elif function_name == "generate_summary":
    #    summary = generate_summary(**arguments)
    #    return summary  # Return structured JSON directly

    # Handle sentiment analysis
    elif function_name == "analyze_sentiment_service":
        return analyze_sentiment_service(**arguments)

    # Handle audio summary generation
    elif function_name == "generate_audio_summary":
        return generate_audio_summary(**arguments)

    # Handle final JSON return
    elif function_name == "return_final_output":
        return json.dumps(arguments, indent=4)  # Ensure JSON is returned properly

    return None  # Handle invalid function names gracefully


def run_agent(input_text: str):
    """Executes the agent step-by-step and ensures controlled function execution."""
    prompt = f"""
    You are an AI assistant that summarizes YouTube videos.
    provide a structured summary strictly in JSON format with these keys:
    - "core_message": (Main idea of the video)
    - "storyline": (Plot or storyline, if applicable)
    - "key_takeaways": (List of main lessons or takeaways)
    - "sentiments": (Emotional tone of the video)
    - "notable_characters": (List of notable characters in the video)
    - "audio_file": (Filename of the generated audio summary)
    - "video_link": (Link to video)      
    """
    messages = [
        {"role": "system","content": prompt},
        {"role": "user", "content": input_text}
    ]
    
    while True:
        # Get the agent's response
        response_str = llm_agent.invoke(messages)

        #try:
            # ✅ Ensure response is JSON
        #    response = json.loads(response_str)

        #except json.JSONDecodeError:
        #    print("Error: LLM did not return JSON. Returning raw response for debugging.")
        #    return {"error": "Invalid response from AI", "response": response_str}  # Return raw response for debugging

        # ✅ Check if AI called a function
        function_call = response_str.get("tool_call", None)

        if function_call:
            function_name = function_call.get("name", "")
            arguments = function_call.get("arguments", {})

            print(f"Agent is calling function: {function_name} with arguments {arguments}")

            # Execute the function safely
            function_result = execute_function_call(function_name, arguments, messages)

            # ✅ Ensure function result is valid JSON
            if isinstance(function_result, dict):
                messages.append({"role": "assistant", "content": json.dumps(function_result, indent=4)})
            elif isinstance(function_result, str):
                try:
                    parsed_result = json.loads(function_result)  # Ensure valid JSON
                    messages.append({"role": "assistant", "content": json.dumps(parsed_result, indent=4)})
                except json.JSONDecodeError:
                    messages.append({"role": "assistant", "content": json.dumps({"error": "Invalid function result"})})

        else:
            print("Final Response:", response_str)
            return response_str  # Stop when there's no function call



# ✅ Define input schema
class VideoRequest(BaseModel):
    video_title: str


@app.post("/summarize")
async def summarize_video(request: VideoRequest):
    """Summarizes a YouTube video given its title."""
    video_title = request.video_title  # Extract from validated input not "notable_characters": (List of notable characters in the video)
    try:
        response = run_agent(video_title)
        summary = extract_summary_data(response)
        return summary

    except json.JSONDecodeError as e:
        raise HTTPException(status_code=500, detail=f"Failed to parse AI response: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

# ✅ Run server if executed directly
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
