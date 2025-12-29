import pymongo
import time
import random
from datetime import datetime, timezone

# MongoDB Connection
MONGO_URI = "mongodb+srv://ashirimi1019:hacklytics@cluster0.tqlrt.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
client = pymongo.MongoClient(MONGO_URI)

# Select Database & Collection
db = client["CervicalDilationDB"]
collection = db["SensorReadings"]

# Function to generate and insert sensor data in real time
def log_sensor_data():
    while True:
        # Simulate sensor readings
        sensor_data = {
            "timestamp": datetime.now(timezone.utc),
            "pressure_mmHg": round(random.uniform(10, 50), 2),  # Random pressure
            "stretch_mm": round(random.uniform(0, 10), 2),  # Random stretch
            "temperature_C": round(random.uniform(36.5, 37.5), 2)  # Random temperature
        }

        # Insert data into MongoDB
        collection.insert_one(sensor_data)

        # Print to confirm
        print("Inserted:", sensor_data)

        # Wait 5 seconds before inserting next reading
        time.sleep(5)

# Run the logger
log_sensor_data()
