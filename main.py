from pathlib import Path
from fastapi import FastAPI, File, UploadFile, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

import numpy as np
from PIL import Image, ImageOps
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

BASE_DIR = Path(__file__).resolve().parent

app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

MODEL_PATH = BASE_DIR / "model" / "potato_leaf_model_1.keras"
CLASS_NAMES_PATH = BASE_DIR / "class_names.json"

with open(CLASS_NAMES_PATH, "r", encoding="utf-8") as f:
    class_names = json.load(f)

model = tf.keras.models.load_model(MODEL_PATH, compile=False)

MODEL_INPUT_SHAPE = model.input_shape[1:3]
print(f"Model input size: {MODEL_INPUT_SHAPE}")
print(f"Class names: {class_names}")


# =========================================================
# Function: Predict Image
# =========================================================
def predict_image(image: Image.Image):
    image = ImageOps.exif_transpose(image)
    image = image.convert("RGB")
    image = image.resize(MODEL_INPUT_SHAPE)

    img_array = tf.keras.preprocessing.image.img_to_array(image)
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array, verbose=0)
    scores = np.squeeze(predictions)

    predicted_index = int(np.argmax(scores))
    confidence = round(float(np.max(scores) * 100), 2)

    if predicted_index >= len(class_names):
        raise ValueError(
            f"Model output classes = {len(scores)}, but class_names = {len(class_names)}"
        )

    predicted_class = class_names[predicted_index]

    if confidence < 96:
        predicted_class = "Invalid image / Not a clear potato leaf image"

    return predicted_class, confidence


# =========================================================
# Function: Confidence Status
# =========================================================
def get_confidence_status(confidence: float) -> str:
    if confidence >= 98:
        return "Very Confident"

    if confidence >= 97:
        return "Good Confidence"

    if confidence < 96:
        predicted_class = "Invalid image / Not a clear potato leaf image"

    return "Invalid image or unclear potato leaf. Please upload a clear potato leaf image."


# =========================================================
# Function: Run Prediction
# =========================================================
async def run_prediction(file: UploadFile):
    allowed_types = ["image/jpeg", "image/jpg", "image/png", "image/webp"]

    if file.content_type not in allowed_types:
        return {
            "ok": False,
            "prediction": None,
            "confidence": None,
            "status": None,
            "error": "Only JPG, JPEG, PNG, or WEBP images are allowed.",
            "uploaded_filename": file.filename,
        }

    contents = await file.read()

    if not contents:
        return {
            "ok": False,
            "prediction": None,
            "confidence": None,
            "status": None,
            "error": "The uploaded file is empty.",
            "uploaded_filename": file.filename,
        }

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
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={}
    )


@app.post("/predict", response_class=HTMLResponse)
async def predict(request: Request, file: UploadFile = File(...)):
    result = await run_prediction(file)

    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={**result}
    )


@app.post("/predict-json")
async def predict_json(file: UploadFile = File(...)):
    result = await run_prediction(file)
    return JSONResponse(result)