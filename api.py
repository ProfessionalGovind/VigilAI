import io
import cv2
import numpy as np
import asyncio 
import os
import time
from PIL import Image
from typing import List, Dict, Any
from fastapi import FastAPI, File, UploadFile, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from sqlmodel import Session # NEW: Import Session for database work

# --- Import my local modules ---
from src.yolov8_detector import YOLOv8Detector
from src.faster_rcnn_detector import FasterRCNNDetector
from db.database import create_db_and_tables, get_session # NEW: Database connection functions
from db.database_models import VigilEvent # NEW: My database model (VigilEvent)


# --- CONFIGURATION & APP SETUP ---
MAX_IMAGES = 5
MAX_VIDEOS = 2
ALLOWED_IMAGE_TYPES = ["image/jpeg", "image/png", "image/webp"]

app = FastAPI(
    title="VigilAI Detection API", 
    description="Scalable API for Real-Time Object Detection and Analytics",
    version="1.0.0"
)

# CORS middleware for frontend connection
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- 1. MODEL LOADING & DB CREATION (Runs ONCE) ---
@app.on_event("startup")
def startup_events():
    """
    This runs once when the API starts. It creates the DB and loads the models.
    """
    # 1. Database Setup: Create the database file and tables if they don't exist.
    create_db_and_tables()
    print("✅ Database tables confirmed/created.")

    # 2. Model Loading: Load the slow detection models into memory.
    print("⏳ Loading detection models... This may take a moment.")
    app.state.detectors = {
        "yolov8": YOLOv8Detector(model_path="models/yolov8n.pt"),
        "frcnn": FasterRCNNDetector(model_path="models/FasterRCNN/faster_rcnn_model.pth")
    }
    print("✅ All detection models loaded and ready!")


# --- HELPER FUNCTION TO PROCESS IMAGES (Asynchronous) ---
# NOTE: This function no longer returns, it just runs the detection.
# We will do the database save in the main endpoint.
async def process_uploaded_image(detector, image_file: UploadFile, model_name: str) -> Dict[str, Any]:
    """
    Helper function to process a single image file concurrently.
    """
    # This is the same logic as before, just wrapped to return results.
    contents = await image_file.read()
    image_stream = io.BytesIO(contents)
    image = Image.open(image_stream).convert("RGB")
    image_np = np.array(image)
    
    detections = detector.detect(image_np)
    
    return {
        "filename": image_file.filename,
        "detections": detections,
        "count": len(detections)
    }


# --- 2. MAIN DETECTION ENDPOINT (With Database Save) ---
@app.post("/detect/{model_name}")
async def run_detection(
    model_name: str, 
    files: List[UploadFile] = File(...),
    # NEW: Inject the database session here as a dependency!
    db: Session = Depends(get_session)
):
    # --- Validation and Processing Logic (Same as before) ---
    if model_name not in app.state.detectors:
        raise HTTPException(status_code=400, detail="Invalid model name selected.")
    
    image_files = [f for f in files if f.content_type in ALLOWED_IMAGE_TYPES]
    video_files = [f for f in files if f.content_type not in ALLOWED_IMAGE_TYPES]

    if len(image_files) == 0:
        raise HTTPException(status_code=400, detail="No valid images sent for detection.")

    detector = app.state.detectors[model_name]
    image_tasks = [process_uploaded_image(detector=detector, image_file=f, model_name=model_name) for f in image_files]
    image_results = await asyncio.gather(*image_tasks)


    # --- NEW: DATABASE LOGIC ---
    total_persons = 0
    # Loop through the results to save data and get a final count
    for result in image_results:
        if result.get("status") != "failed":
            total_persons += result['count']
            
            # Create a new event record
            event = VigilEvent(
                model_used=model_name,
                video_source="API_UPLOAD", # Hardcoded source for now
                total_detections=result['count'],
                person_count=result['count'],
                is_anomaly=result['count'] > 5, # Simple anomaly check: if more than 5 people
                timestamp=datetime.utcnow()
            )
            
            # Save it to the database!
            db.add(event)
            db.commit() # Commit each record
            db.refresh(event)
            
    # --- Final Response ---
    return {
        "status": "success",
        "model_used": model_name,
        "total_persons_detected": total_persons,
        "image_processing_summary": image_results,
        "note": "Analytics data was saved to PostgreSQL (SQLite for dev) database."
    }


# --- TEST ENDPOINT ---
@app.get("/")
def read_root():
    # Showing how many models I've loaded.
    model_count = len(app.state.detectors)
    return {"message": "VigilAI API is running!", "models_loaded": model_count}