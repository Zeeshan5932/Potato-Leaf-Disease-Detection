from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import tensorflow as tf
from PIL import Image
import numpy as np
import io

# Load the trained model
model = tf.keras.models.load_model("./model/efficientnetb3_potato_model.h5")

# Replace with your actual class names
class_names = ["Early_blight", "Late_blight", "Healthy"]

app = FastAPI(title="Potato Disease Detection API")

# Home route
@app.get("/")
def home():
    return {"message": "Welcome to Potato Disease Detection API"}

# Prediction route
@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    try:
        # Read image bytes
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        
        # Resize to model input size
        image = image.resize((300, 300))  # EfficientNetB3 default is 300x300
        img_array = np.array(image) / 255.0  # normalize
        img_array = np.expand_dims(img_array, axis=0)
        
        # Predict
        predictions = model.predict(img_array)
        predicted_class = class_names[np.argmax(predictions)]
        confidence = float(np.max(predictions))
        
        return JSONResponse({
            "class": predicted_class,
            "confidence": confidence
        })
        
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=400)