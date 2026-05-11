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
MODEL_PATHS = [
    BASE_DIR / "model" / "potato_disease_model.keras",
    BASE_DIR / "model" / "potato_disease_model.h5",
]
CLASS_NAMES_PATH = BASE_DIR / "class_names.json"
IMAGE_SIZE = 256

MODEL_PATH = next((path for path in MODEL_PATHS if path.exists()), None)

with open(CLASS_NAMES_PATH, "r", encoding="utf-8") as f:
    class_names = json.load(f)

tf = None
model = None
model_load_error = None


def load_model_if_needed():
    global tf, model, model_load_error

    if model is not None:
        return

    if model_load_error is not None:
        raise RuntimeError(model_load_error)

    if MODEL_PATH is None:
        model_load_error = (
            "No trained model found. Expected one of: "
            + ", ".join(str(path) for path in MODEL_PATHS)
        )
        raise RuntimeError(model_load_error)

    try:
        import tensorflow as tensorflow_lib

        tf = tensorflow_lib
        model = tf.keras.models.load_model(MODEL_PATH)
    except Exception as exc:
        model_load_error = str(exc)
        raise RuntimeError(model_load_error) from exc


def predict_image(image: Image.Image):
    load_model_if_needed()

    image = image.convert("RGB")
    image = image.resize((IMAGE_SIZE, IMAGE_SIZE))

    img_array = tf.keras.preprocessing.image.img_to_array(image)
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array)
    scores = np.squeeze(predictions)
    predicted_index = int(np.argmax(scores))

    if predicted_index >= len(class_names):
        raise ValueError("Model output does not match the configured class names.")

    predicted_class = class_names[predicted_index]
    confidence = round(float(np.max(scores) * 100), 2)

    return predicted_class, confidence


def get_confidence_status(confidence: float) -> str:
    if confidence >= 90:
        return "Very Confident"
    if confidence >= 80:
        return "Good Confidence"
    if confidence >= 70:
        return "Acceptable Confidence"
    return "Low Confidence - Please upload a clearer potato leaf image"


async def run_prediction(file: UploadFile):
    if not file.content_type or not file.content_type.startswith("image/"):
        return {
            "ok": False,
            "prediction": None,
            "confidence": None,
            "error": "Please upload a valid image file.",
            "uploaded_filename": None,
            "model_name": MODEL_PATH.name if MODEL_PATH else "Unavailable",
        }

    contents = await file.read()
    if not contents:
        return {
            "ok": False,
            "prediction": None,
            "confidence": None,
            "error": "The uploaded file is empty.",
            "uploaded_filename": None,
            "model_name": MODEL_PATH.name if MODEL_PATH else "Unavailable",
        }

    try:
        image = Image.open(io.BytesIO(contents))
        prediction, confidence = predict_image(image)
        status = get_confidence_status(confidence)

        return {
            "ok": True,
            "prediction": prediction,
            "confidence": confidence,
            "status": status,
            "error": None,
            "uploaded_filename": file.filename,
            "model_name": MODEL_PATH.name if MODEL_PATH else "Unavailable",
        }
    except Exception as exc:
        return {
            "ok": False,
            "prediction": None,
            "confidence": None,
            "status": None,
            "error": f"Error: {str(exc)}",
            "uploaded_filename": file.filename,
            "model_name": MODEL_PATH.name if MODEL_PATH else "Unavailable",
        }


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(request=request, name="index.html", context={})


@app.post("/predict", response_class=HTMLResponse)
async def predict(request: Request, file: UploadFile = File(...)):
    result = await run_prediction(file)
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={**result},
    )


@app.post("/predict-json")
async def predict_json(file: UploadFile = File(...)):
    result = await run_prediction(file)
    return JSONResponse(result)