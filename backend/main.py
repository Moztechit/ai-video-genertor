import os
import asyncio
import requests
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import edge_tts
import replicate

app = FastAPI(title="AI Talking Avatar Generator (No-Card Version)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class GenerationRequest(BaseModel):
    project_id: str
    script_text: str
    voice_name: str = "en-US-AriaNeural"  # Default Microsoft Edge neural voice
    image_url: str

PROJECT_DATABASE = {}

async def generate_audio_edge(text: str, voice: str, output_path: str):
    communicate = edge_tts.Communicate(text, voice)
    await communicate.save(output_path)

def process_video_pipeline(project_id: str, script_text: str, voice_name: str, image_url: str):
    try:
        # Step 1: Free Edge Text-to-Speech
        PROJECT_DATABASE[project_id] = {"status": "PROCESSING_VOICE", "progress": 20}
        
        temp_audio_file = f"/tmp/{project_id}.mp3"
        
        # Run async edge-tts inside sync background task
        asyncio.run(generate_audio_edge(script_text, voice_name, temp_audio_file))
        
        with open(temp_audio_file, "rb") as f:
            audio_bytes = f.read()

        # Helper storage function simulation (In production, push audio_bytes to S3/Cloudinary)
        audio_url = upload_to_temp_storage(audio_bytes, "audio.mp3")

        # Step 2: Fal.ai Image-to-Video (Wan 2.1)
        PROJECT_DATABASE[project_id].update({"status": "PROCESSING_VIDEO", "progress": 50, "audio_url": audio_url})
        
        fal_key = os.getenv("FAL_KEY")
        fal_res = requests.post(
            "https://fal.run/fal-ai/wan-i2v",
            headers={"Authorization": f"Key {fal_key}"},
            json={"image_url": image_url, "prompt": "Subtle expressive speaking movement, high quality portrait"}
        )
        if fal_res.status_code != 200:
            raise Exception(f"Fal.ai error: {fal_res.text}")
        
        base_video_url = fal_res.json().get("video", {}).get("url")

        # Step 3: Replicate Lip-Sync
        PROJECT_DATABASE[project_id].update({"status": "PROCESSING_SYNCHRONIZING", "progress": 80, "base_video_url": base_video_url})

        output = replicate.run(
            "sync/lipsync-pro:latest",
            input={"face": base_video_url, "audio": audio_url}
        )
        final_video_url = output

        PROJECT_DATABASE[project_id] = {
            "status": "COMPLETED",
            "progress": 100,
            "final_video_url": final_video_url
        }

    except Exception as e:
        PROJECT_DATABASE[project_id] = {
            "status": "FAILED",
            "progress": 0,
            "error": str(e)
        }

def upload_to_temp_storage(file_bytes: bytes, filename: str):
    # Replace this with real AWS S3 / Cloudinary upload logic returning a public URL
    return "https://example.com/mock-temp-audio.mp3"

@app.post("/api/generate")
async def start_generation(data: GenerationRequest, background_tasks: BackgroundTasks):
    PROJECT_DATABASE[data.project_id] = {"status": "PENDING", "progress": 10}
    background_tasks.add_task(
        process_video_pipeline, 
        data.project_id, 
        data.script_text, 
        data.voice_name, 
        data.image_url
    )
    return {"message": "Generation started", "project_id": data.project_id}

@app.get("/api/status/{project_id}")
async def get_status(project_id: str):
    if project_id not in PROJECT_DATABASE:
        raise HTTPException(status_code=404, detail="Project not found")
    return PROJECT_DATABASE[project_id]
