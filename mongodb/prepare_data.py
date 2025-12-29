import pymongo
import pandas as pd

# MongoDB Connection
MONGO_URI = "mongodb+srv://ashirimi1019:hacklytics@cluster0.tqlrt.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
client = pymongo.MongoClient(MONGO_URI)

# Select Database & Collection
db = client["CervicalDilationDB"]
collection = db["SensorReadings"]

# Retrieve Data
data = list(collection.find().sort("timestamp", 1))

# Convert to DataFrame
df = pd.DataFrame(data)

# Convert timestamp to datetime
df["timestamp"] = pd.to_datetime(df["timestamp"])

# Drop unnecessary MongoDB ID column
df.drop(columns=["_id"], inplace=True)

# Simulate dilation stages based on stretch values
def categorize_dilation(stretch):
    if stretch < 3:
        return "Early Labor"
    elif 3 <= stretch < 7:
        return "Active Labor"
    else:
        return "Transition Stage"

df["dilation_stage"] = df["stretch_mm"].apply(categorize_dilation)

# Save processed data
df.to_csv("processed_sensor_data.csv", index=False)

print("Data preprocessing complete! Saved as 'processed_sensor_data.csv'.")
