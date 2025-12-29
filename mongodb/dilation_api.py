import os
import logging
from dotenv import load_dotenv
from pymongo import MongoClient
from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
from typing import List
from datetime import datetime
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from bson.binary import Binary

# ✅ Load environment variables
load_dotenv()
MONGO_URI = os.getenv("MONGO_URI")

if not MONGO_URI:
    raise ValueError("MongoDB URI not found. Make sure .env is configured correctly!")

# ✅ Setup Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ✅ Connect to MongoDB
try:
    client = MongoClient(MONGO_URI)
    db = client["cervical_dilation"]
    history_collection = db["dilation_history"]
    image_collection = db["ultrasound_images"]
    logger.info("✅ Connected to MongoDB successfully")
except Exception as e:
    logger.error(f"❌ MongoDB Connection Error: {e}")
    raise HTTPException(status_code=500, detail="Database connection failed.")

# ✅ Load DeepSeek Model
model_name = "deepseek-ai/deepseek-llm-7b-chat"
device = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"Using device: {device}")

offload_path = "./offload_model"
os.makedirs(offload_path, exist_ok=True)

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    device_map="auto",
    offload_folder=offload_path
)

# ✅ FastAPI App
app = FastAPI()

# ✅ Define Input Data Model
class SensorData(BaseModel):
    pressure_mmHg: float
    stretch_mm: float
    temperature_C: float

# ✅ Prediction API Route
@app.post("/predict")
def predict_dilation(data: SensorData):
    """Predict the cervical dilation stage based on sensor data."""
    logger.info("🔄 Processing prediction request")

    # 🔹 New and Improved Prompt (Forces Strict Output)
    prompt = f"""
    A pregnant patient has these sensor readings:
    - Cervical Pressure: {data.pressure_mmHg} mmHg
    - Cervical Stretch: {data.stretch_mm} mm
    - Body Temperature: {data.temperature_C} °C

    Based on medical knowledge, determine the current cervical dilation stage:
    - Early Labor
    - Active Labor
    - Transition Stage

    ONLY return the stage name followed by a short next-step recommendation for the patient. 
    DO NOT explain what cervical dilation is.  
    Example Response Format:
    "Active Labor. You should prepare to go to the hospital soon."
    """

    try:
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        outputs = model.generate(**inputs, max_length=50, temperature=0)  # 🔹 Strict, deterministic response
        response_text = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
    except Exception as e:
        logger.error(f"❌ Model prediction error: {e}")
        raise HTTPException(status_code=500, detail="AI model prediction failed.")

    # ✅ Ensure the AI returns one of the three stages
    valid_stages = ["Early Labor", "Active Labor", "Transition Stage"]
    predicted_stage = next((stage for stage in valid_stages if stage in response_text), "Unknown")

    # 🔹 Force a Clear Response Format
    response_mapping = {
        "Early Labor": "Early Labor. Monitor contractions and prepare for active labor soon.",
        "Active Labor": "Active Labor. You should prepare to go to the hospital soon.",
        "Transition Stage": "Transition Stage. Go to the hospital immediately."
    }
    final_response = response_mapping.get(predicted_stage, "Unknown. Please consult a doctor.")

    # ✅ Save to MongoDB
    prediction_record = {
        "timestamp": datetime.utcnow(),
        "input_data": data.dict(),
        "prediction": final_response
    }
    history_collection.insert_one(prediction_record)

    logger.info(f"✅ Prediction saved to MongoDB: {final_response}")

    return {"dilation_prediction": final_response}

# ✅ Upload Multiple Ultrasound Images
@app.post("/upload_ultrasounds")
async def upload_ultrasound_images(files: List[UploadFile] = File(...)):
    """Upload multiple ultrasound images and store them in MongoDB."""
    
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded.")

    uploaded_files = []
    
    for file in files:
        file_content = await file.read()
        image_document = {
            "filename": file.filename,
            "content_type": file.content_type,
            "upload_time": datetime.utcnow(),
            "data": Binary(file_content)  # Store as BSON Binary
        }
        image_collection.insert_one(image_document)
        uploaded_files.append(file.filename)

    logger.info(f"✅ Uploaded {len(uploaded_files)} images: {uploaded_files}")

    return {"message": "Images uploaded successfully", "filenames": uploaded_files}

# ✅ List Uploaded Ultrasound Images
@app.get("/ultrasound_images")
def list_ultrasound_images():
    """List all uploaded ultrasound images."""
    images = list(image_collection.find({}, {"_id": 0, "filename": 1, "upload_time": 1}))
    if not images:
        raise HTTPException(status_code=404, detail="No ultrasound images found.")

    return {"images": images}

# ✅ Retrieve Specific Ultrasound Image
@app.get("/ultrasound_image/{filename}")
def get_ultrasound_image(filename: str):
    """Retrieve an ultrasound image by filename."""
    image = image_collection.find_one({"filename": filename}, {"_id": 0, "data": 1, "content_type": 1})
    if not image:
        raise HTTPException(status_code=404, detail="Image not found.")

    return {"filename": filename, "content_type": image["content_type"], "data": image["data"]}

# ✅ Delete a Specific Ultrasound Image
@app.delete("/delete_ultrasound/{filename}")
def delete_ultrasound_image(filename: str):
    """Delete an ultrasound image from MongoDB."""
    result = image_collection.delete_one({"filename": filename})
    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="Image not found.")

    logger.info(f"✅ Deleted image: {filename}")
    return {"message": "Image deleted successfully", "filename": filename}

# ✅ Delete Prediction History
@app.delete("/delete_dilation_history")
def delete_dilation_history():
    """Delete all dilation history records."""
    result = history_collection.delete_many({})
    logger.info(f"✅ Deleted {result.deleted_count} records from history.")

    return {"message": f"Deleted {result.deleted_count} records."}

# ✅ Get MongoDB Status
@app.get("/mongodb_status")
def mongodb_status():
    """Check if MongoDB connection is alive."""
    try:
        client.admin.command('ping')
        return {"status": "MongoDB connection is active"}
    except Exception as e:
        logger.error(f"❌ MongoDB Status Check Failed: {e}")
        raise HTTPException(status_code=500, detail="MongoDB connection failed.")

# ✅ AI Model Status Check
@app.get("/ai_status")
def ai_status():
    """Check if the AI model is loaded and ready."""
    try:
        test_prompt = "Check AI status."
        test_inputs = tokenizer(test_prompt, return_tensors="pt").to(device)
        _ = model.generate(**test_inputs, max_length=10)
        return {"status": "AI model is ready"}
    except Exception as e:
        logger.error(f"❌ AI Model Status Check Failed: {e}")
        raise HTTPException(status_code=500, detail="AI model is not ready.")


