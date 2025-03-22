from fastapi import FastAPI, UploadFile, Form
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import os

# from src.ai_app import analyze_video
from ai_app2 import analyze_video_with_function_calls

# Initialize FastAPI app
app = FastAPI(title="AI Youtube Video Summary API", version="1.0")

# Enable CORS for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class VideoRequestQuery(BaseModel):
    user_query: str
    generate_tts: bool


# Health check endpoint
@app.get("/")
def read_root():
    return {"message": "AI Youtube Video Summary API is running."}


#@app.post("/summarize")
#def get_video_summary(video_request: VideoRequestQuery):
#    analysis = analyze_video(video_request.user_query, video_request.generate_tts)
#    return JSONResponse(analysis)


@app.post("/summarize2")
async def summarize_video(video_request: VideoRequestQuery):
    analysis = analyze_video_with_function_calls(
        video_request.user_query, video_request.generate_tts
    )
    return JSONResponse(analysis)


# Endpoint to download audio summary
@app.get("/download-audio/{file_name}")
async def download_file(file_name: str):
    file_path = f"static/{file_name}"
    return FileResponse(file_path)
