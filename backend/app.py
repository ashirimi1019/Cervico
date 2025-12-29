from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pymongo import MongoClient
from datetime import datetime
import os
import sys
import numpy as np
from PIL import Image
import io
import base64
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Debug prints
print("Current working directory:", os.getcwd())
print("Environment variables:", os.environ.keys())
print("MONGO_URI value:", os.getenv('MONGO_URI'))

# Add AI model directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'ai'))
from src.predict import predict_dilation

app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # React app URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# MongoDB Atlas connection
MONGO_URI = os.getenv('MONGO_URI')
if not MONGO_URI:
    raise ValueError("No MONGO_URI found in environment variables")

try:
    client = MongoClient(MONGO_URI)
    db = client.cervico
    ultrasounds = db.ultrasounds
    print("Connected to MongoDB Atlas successfully!")
except Exception as e:
    print(f"Failed to connect to MongoDB Atlas: {e}")

# Define model path
MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'ai', 'models', 'best_model.pt')
if not os.path.exists(MODEL_PATH):
    raise ValueError(f"Model file not found at {MODEL_PATH}")
else:
    print(f"Found AI model at: {MODEL_PATH}")

@app.post("/process-ultrasound")
async def process_ultrasound(file: UploadFile = File(...)):
    try:
        # Read and process the image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        
        # Save image temporarily
        temp_path = "temp_ultrasound.jpg"
        image.save(temp_path)
        
        try:
            # Get predictions from AI model
            class_pred, precise_pred = predict_dilation(
                image_path=temp_path,
                model_path=MODEL_PATH,
                device='cpu'  # Use CPU for prediction
            )
            print(f"AI Prediction successful - Class: {class_pred:.1f}cm, Precise: {precise_pred:.2f}cm")
        except Exception as e:
            print(f"AI prediction error: {str(e)}")
            raise HTTPException(status_code=500, detail=f"AI model prediction failed: {str(e)}")

        # Convert image to base64 for storage
        buffered = io.BytesIO()
        image.save(buffered, format="JPEG")
        image_base64 = base64.b64encode(buffered.getvalue()).decode()
        
        # Store in MongoDB
        result = ultrasounds.insert_one({
            "timestamp": datetime.utcnow(),
            "image": image_base64,
            "class_prediction": float(class_pred),
            "precise_prediction": float(precise_pred),
            "filename": file.filename
        })
        
        # Clean up temp file
        if os.path.exists(temp_path):
            os.remove(temp_path)
        
        return {
            "success": True,
            "id": str(result.inserted_id),
            "class_prediction": float(class_pred),
            "precise_prediction": float(precise_pred),
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        if os.path.exists("temp_ultrasound.jpg"):
            os.remove("temp_ultrasound.jpg")
        print(f"Error processing ultrasound: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/ultrasound/{ultrasound_id}")
async def get_ultrasound(ultrasound_id: str):
    try:
        from bson.objectid import ObjectId
        
        # Fetch ultrasound data from MongoDB
        result = ultrasounds.find_one({"_id": ObjectId(ultrasound_id)})
        
        if result:
            return {
                "success": True,
                "data": {
                    "id": str(result["_id"]),
                    "class_prediction": result["class_prediction"],
                    "precise_prediction": result["precise_prediction"],
                    "timestamp": result["timestamp"].isoformat(),
                    "filename": result["filename"],
                    "image": result["image"]  # Base64 encoded image
                }
            }
        else:
            raise HTTPException(status_code=404, detail="Ultrasound not found")
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
