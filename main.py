from pathlib import Path
from fastapi import FastAPI, File, UploadFile, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

import numpy as np
from PIL import Image
import json
import io
import tensorflow as tf

app = FastAPI(title="Potato Leaf Disease Detection")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model" / "efficientnetb3_potato_model.h5"
CLASS_NAMES_PATH = BASE_DIR / "class_names.json"

with open(CLASS_NAMES_PATH, "r", encoding="utf-8") as f:
    class_names = json.load(f)

model = tf.keras.models.load_model(MODEL_PATH)
MODEL_INPUT_SHAPE = model.input_shape[1:3]  # e.g., (300,300)
print(f"Model input size: {MODEL_INPUT_SHAPE}")

# =========================================================
# Function: Predict Image
# =========================================================
def predict_image(image: Image.Image):
    image = image.convert("RGB")
    # Resize image to model's expected input size
    image = image.resize(MODEL_INPUT_SHAPE)

    img_array = tf.keras.preprocessing.image.img_to_array(image)
    img_array = np.expand_dims(img_array, axis=0) / 255.0  # normalize

    predictions = model.predict(img_array)
    scores = np.squeeze(predictions)
    predicted_index = int(np.argmax(scores))

    if predicted_index >= len(class_names):
        raise ValueError("Model output does not match the configured class names.")

    predicted_class = class_names[predicted_index]
    confidence = round(float(np.max(scores) * 100), 2)

    return predicted_class, confidence

# =========================================================
# Function: Confidence Status
# =========================================================
def get_confidence_status(confidence: float) -> str:
    if confidence >= 90:
        return "Very Confident"
    if confidence >= 80:
        return "Good Confidence"
    if confidence >= 70:
        return "Acceptable Confidence"
    return "Low Confidence - Please upload a clearer potato leaf image"

# =========================================================
# Function: Run Prediction
# =========================================================
async def run_prediction(file: UploadFile):
    if not file.content_type or not file.content_type.startswith("image/"):
        return {"ok": False, "prediction": None, "confidence": None,
                "error": "Please upload a valid image file.", "uploaded_filename": None}

    contents = await file.read()
    if not contents:
        return {"ok": False, "prediction": None, "confidence": None,
                "error": "The uploaded file is empty.", "uploaded_filename": None}

    try:
        image_obj = Image.open(io.BytesIO(contents))
        prediction, confidence = predict_image(image_obj)
        status = get_confidence_status(confidence)

        return {
            "ok": True,
            "prediction": prediction,
            "confidence": confidence,
            "status": status,
            "uploaded_filename": file.filename,
        }
    except Exception as exc:
        return {
            "ok": False,
            "prediction": None,
            "confidence": None,
            "status": None,
            "error": f"Error: {str(exc)}",
            "uploaded_filename": file.filename,
        }

# =========================================================
# Routes
# =========================================================
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(request=request, name="index.html", context={})

@app.post("/predict", response_class=HTMLResponse)
async def predict(request: Request, file: UploadFile = File(...)):
    result = await run_prediction(file)
    return templates.TemplateResponse(request=request, name="index.html", context={**result})

@app.post("/predict-json")
async def predict_json(file: UploadFile = File(...)):
    result = await run_prediction(file)
    return JSONResponse(result)