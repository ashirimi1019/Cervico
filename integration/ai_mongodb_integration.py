import os
import logging
from datetime import datetime, timezone
from typing import Dict, Any, Optional
import pymongo
from dotenv import load_dotenv
from PIL import Image
import io

# Import AI components
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from ai.src.predict import predict_dilation

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
MONGO_URI = os.getenv("MONGO_URI")
if not MONGO_URI:
    raise ValueError("MongoDB URI not found in environment variables")

class CervicoIntegration:
    def __init__(self):
        """Initialize MongoDB connection and AI model"""
        # Connect to MongoDB
        self.client = pymongo.MongoClient(MONGO_URI)
        self.db = self.client["cervical_dilation"]
        
        # Collections
        self.predictions = self.db["dilation_predictions"]
        self.images = self.db["ultrasound_images"]
        self.sensor_data = self.db["sensor_readings"]
        
        # AI model path
        self.model_path = os.path.join(os.path.dirname(__file__), '../ai/models/best_model.pt')
        
        logger.info("✅ Integration initialized successfully")

    def process_ultrasound_image(self, image_data: bytes, filename: str) -> Dict[str, Any]:
        """
        Process an ultrasound image through the AI model and store results
        
        Args:
            image_data: Binary image data
            filename: Name of the image file
            
        Returns:
            Dictionary containing prediction results
        """
        # Save image temporarily
        temp_path = f"temp_{filename}"
        with open(temp_path, 'wb') as f:
            f.write(image_data)
        
        try:
            # Get AI prediction
            class_pred, precise_pred = predict_dilation(
                image_path=temp_path,
                model_path=self.model_path
            )
            
            # Store image and prediction in MongoDB
            prediction_doc = {
                "timestamp": datetime.now(timezone.utc),
                "filename": filename,
                "class_prediction": float(class_pred),
                "precise_prediction": float(precise_pred),
                "image_data": image_data
            }
            
            # Store in MongoDB
            self.predictions.insert_one(prediction_doc)
            
            logger.info(f"✅ Processed and stored prediction for {filename}")
            return {
                "filename": filename,
                "class_prediction": float(class_pred),
                "precise_prediction": float(precise_pred)
            }
            
        finally:
            # Cleanup temporary file
            if os.path.exists(temp_path):
                os.remove(temp_path)

    def correlate_sensor_data(self, image_prediction: Dict[str, Any]) -> Dict[str, Any]:
        """
        Correlate image prediction with recent sensor data
        
        Args:
            image_prediction: Dictionary containing image prediction results
            
        Returns:
            Dictionary with combined image and sensor data
        """
        # Get most recent sensor reading
        recent_sensor = self.sensor_data.find_one(
            sort=[("timestamp", pymongo.DESCENDING)]
        )
        
        if recent_sensor:
            return {
                **image_prediction,
                "sensor_data": {
                    "pressure_mmHg": recent_sensor["pressure_mmHg"],
                    "stretch_mm": recent_sensor["stretch_mm"],
                    "temperature_C": recent_sensor["temperature_C"],
                    "sensor_timestamp": recent_sensor["timestamp"]
                }
            }
        return image_prediction

    def get_patient_history(self, start_date: datetime, end_date: datetime) -> Dict[str, Any]:
        """
        Retrieve patient history between dates
        
        Args:
            start_date: Start datetime
            end_date: End datetime
            
        Returns:
            Dictionary containing prediction and sensor history
        """
        date_filter = {
            "timestamp": {
                "$gte": start_date,
                "$lte": end_date
            }
        }
        
        predictions = list(self.predictions.find(date_filter, {"image_data": 0}))
        sensor_readings = list(self.sensor_data.find(date_filter))
        
        return {
            "predictions": predictions,
            "sensor_readings": sensor_readings
        }

    def cleanup_old_data(self, days_to_keep: int = 30):
        """
        Clean up data older than specified days
        
        Args:
            days_to_keep: Number of days of data to retain
        """
        cutoff_date = datetime.now(timezone.utc) - timedelta(days=days_to_keep)
        
        # Remove old predictions and images
        self.predictions.delete_many({"timestamp": {"$lt": cutoff_date}})
        self.sensor_data.delete_many({"timestamp": {"$lt": cutoff_date}})
        
        logger.info(f"✅ Cleaned up data older than {days_to_keep} days")

if __name__ == "__main__":
    # Example usage
    integration = CervicoIntegration()
    
    # Process a test image
    with open("test_image.jpg", "rb") as f:
        result = integration.process_ultrasound_image(f.read(), "test_image.jpg")
        
    # Correlate with sensor data
    combined_result = integration.correlate_sensor_data(result)
    print("Combined Result:", combined_result)
