Cervico - AI-Powered Cervical Dilation Tracker

Overview

Cervico is an AI-powered application designed to monitor and estimate cervical dilation progression using transvaginal ultrasound technology. This innovative tool aims to provide non-invasive, real-time cervical dilation tracking during labor, reducing the need for frequent manual cervical checks.

Features

AI-Powered Cervical Dilation Estimation: Uses convolutional neural networks (CNN) to analyze cervical dilation images.

Tampon-Style Ultrasound Patch: A wearable, transvaginal ultrasound device that provides continuous monitoring.

Real-Time Mobile App Interface: Displays dilation measurements, labor progress trends, and alerts for active labor.

Early Stage Prototype with Diagrams: Initial model training using medical diagrams before integrating real ultrasound data.

Predictive Labor Tracking: Combines contraction frequency and dilation trends to predict labor progression.

Epidural Timing Alerts: Notifies patients when it is the optimal time to receive an epidural based on dilation and contraction patterns.

How It Works

Data Input

Users wear the Cervico Patch as instructed, which captures real-time cervical images.

Initially, AI is trained using diagram-based cervical dilation images and will later incorporate real ultrasound data.

AI Processing

The AI model analyzes images to detect cervical opening size (in cm).

Predictions are refined with real ultrasound training data over time.

User Alerts & Insights

The mobile app notifies users when dilation reaches active labor (~4 cm).

**Dependency Report for FastAPI-based Backend**

### Overview

This report provides an analysis of the dependencies used in the provided FastAPI-based backend code. Each dependency is explained with its purpose and relevance in the project.

---

### Dependencies Used and Their Purpose

#### 1. **FastAPI**

- **Modules Used**: `FastAPI`, `File`, `UploadFile`, `HTTPException`
- **Purpose**: FastAPI is the main web framework used to create the API endpoints for processing ultrasound images and retrieving data from MongoDB.
  - `FastAPI` is used to define the API routes and handle requests.
  - `File` and `UploadFile` enable handling file uploads, such as ultrasound images.
  - `HTTPException` is used to handle and return HTTP errors properly.

#### 2. **fastapi.middleware.cors**

- **Module Used**: `CORSMiddleware`
- **Purpose**: Manages Cross-Origin Resource Sharing (CORS), allowing the frontend (e.g., React app at `http://localhost:3000`) to communicate with the backend.

#### 3. **pymongo**

- **Module Used**: `MongoClient`
- **Purpose**: Establishes a connection with MongoDB Atlas for storing and retrieving ultrasound prediction data.

#### 4. **datetime**

- **Module Used**: `datetime`
- **Purpose**: Handles timestamps for storing ultrasound processing times in MongoDB.

#### 5. **os**

- **Module Used**: `os`
- **Purpose**: Used for interacting with the operating system, such as:
  - Retrieving environment variables (`os.getenv('MONGO_URI')`).
  - Managing file paths for AI model loading.
  - Cleaning up temporary files after processing images.

#### 6. **sys**

- **Module Used**: `sys`
- **Purpose**: Modifies the Python path (`sys.path.append(...)`) to include the AI model directory (`src.predict`) for importing custom modules.

#### 7. **numpy**

- **Module Used**: `numpy`
- **Purpose**: Used for numerical computations, likely within the AI model (`predict_dilation`).

#### 8. **PIL (Pillow)**

- **Module Used**: `Image`
- **Purpose**: Handles image processing, including:
  - Reading ultrasound images.
  - Converting images to different formats.
  - Saving temporary images for AI model processing.

#### 9. **io**

- **Module Used**: `io`
- **Purpose**: Manages in-memory binary stream operations, such as:
  - Converting images to byte streams for MongoDB storage.
  - Reading uploaded images as `BytesIO` objects.

#### 10. **base64**

- **Module Used**: `base64`
- **Purpose**: Converts ultrasound images to Base64 format before storing them in MongoDB.

#### 11. **python-dotenv**

- **Module Used**: `load_dotenv`
- **Purpose**: Loads environment variables (e.g., `MONGO_URI`) from a `.env` file, making configuration more secure and dynamic.

#### 12. **bson**

- **Module Used**: `ObjectId`
- **Purpose**: Retrieves MongoDB documents using ObjectId when fetching ultrasound data.

#### 13. **uvicorn**

- **Module Used**: `uvicorn`
- **Purpose**: Runs the FastAPI application as an ASGI server (`uvicorn.run(...)`).

---

### Conclusion

This backend application primarily leverages FastAPI for API handling, MongoDB for storage, and AI model integration for ultrasound image predictions. Each dependency plays a crucial role in enabling efficient API development, secure data handling, and image processing. Proper management of these dependencies ensures optimal performance and maintainability of the system.



Users receive trend reports and insights on labor progression.

Provides alerts for the best time to receive an epidural, ensuring optimal pain management timing.

