import pymongo
from datetime import datetime, timezone

# MongoDB Connection String
MONGO_URI = "mongodb+srv://ashirimi1019:hacklytics@cluster0.tqlrt.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"

# Connect to MongoDB
client = pymongo.MongoClient(MONGO_URI)

# Create database & collection
db = client["CervicalDilationDB"]
collection = db["SensorReadings"]

# Insert a test document
sample_data = {
    "timestamp": datetime.now(timezone.utc),
    "pressure_mmHg": 30.5,
    "stretch_mm": 5.1,
    "temperature_C": 36.7
}

collection.insert_one(sample_data)

print("Test data inserted successfully!")
