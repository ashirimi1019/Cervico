import pymongo
import pandas as pd
import matplotlib.pyplot as plt

# MongoDB Connection
MONGO_URI = "mongodb+srv://ashirimi1019:hacklytics@cluster0.tqlrt.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
client = pymongo.MongoClient(MONGO_URI)

# Select Database & Collection
db = client["CervicalDilationDB"]
collection = db["SensorReadings"]

# Retrieve Data from MongoDB
data = list(collection.find().sort("timestamp", 1))  # Sort by time ascending

# Convert to Pandas DataFrame
df = pd.DataFrame(data)

# Convert timestamp to datetime format
df["timestamp"] = pd.to_datetime(df["timestamp"])

# Plot Sensor Data
plt.figure(figsize=(10, 5))
plt.plot(df["timestamp"], df["pressure_mmHg"], label="Pressure (mmHg)", marker="o")
plt.plot(df["timestamp"], df["stretch_mm"], label="Stretch (mm)", marker="s")
plt.plot(df["timestamp"], df["temperature_C"], label="Temperature (°C)", marker="^")

# Customize the plot
plt.xlabel("Time")
plt.ylabel("Sensor Values")
plt.title("Cervical Dilation Sensor Data Over Time")
plt.legend()
plt.xticks(rotation=45)
plt.grid()

# Show the plot
plt.show()
