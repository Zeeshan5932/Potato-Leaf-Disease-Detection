# from fastapi import FastAPI, Request, UploadFile, File
# from fastapi.responses import JSONResponse, HTMLResponse
# from fastapi.staticfiles import StaticFiles
# from fastapi.templating import Jinja2Templates

# import tensorflow as tf
# import numpy as np
# from PIL import Image
# import io
# import os

# # =====================================================
# # FastAPI App
# # =====================================================

# app = FastAPI(
#     title="Potato Leaf Disease Detection",
#     version="1.0.0"
# )

# # =====================================================
# # Static & Templates
# # =====================================================

# app.mount("/static", StaticFiles(directory="static"), name="static")

# templates = Jinja2Templates(directory="templates")

# # =====================================================
# # Load Model
# # =====================================================

# MODEL_PATH = os.path.join("model", "potatoes.keras")

# model = tf.keras.models.load_model(MODEL_PATH)

# print("✅ Model Loaded Successfully")

# # =====================================================
# # Classes
# # =====================================================

# CLASS_NAMES = [
#     "Potato___Early_Blight",
#     "Potato___Late_Blight",
#     "Potato___Healthy"
# ]


# def preprocess_image(image: Image.Image):

#     image = image.convert("RGB")

#     image = image.resize((256, 256))

#     img = np.array(image, dtype=np.float32)

#     img = np.expand_dims(img, axis=0)

#     return img


# # =====================================================
# # Home
# # =====================================================

# @app.get("/", response_class=HTMLResponse)
# async def home(request: Request):

#     return templates.TemplateResponse(
#         name="index.html",
#         request=request
#     )


# # =====================================================
# # Prediction API
# # =====================================================

# @app.post("/predict-json")
# async def predict(file: UploadFile = File(...)):

#     try:

#         contents = await file.read()

#         image = Image.open(io.BytesIO(contents))

#         img = preprocess_image(image)

#         prediction = model.predict(img, verbose=0)

#         predicted_index = np.argmax(prediction)

#         predicted_class = CLASS_NAMES[predicted_index]

#         confidence = float(np.max(prediction) * 100)

#         if predicted_class == "Potato___Healthy":

#             status = "Healthy"

#         else:

#             status = "Disease Detected"

#         return JSONResponse(
#             {

#                 "ok": True,

#                 "prediction": predicted_class,

#                 "confidence": round(confidence, 2),

#                 "status": status,

#                 "uploaded_filename": file.filename,

#                 "model_name": "Potato Disease CNN"

#             }
#         )

#     except Exception as e:

#         return JSONResponse(

#             status_code=500,

#             content={

#                 "ok": False,

#                 "error": str(e)

#             }

#         )


# # =====================================================
# # Health Check
# # =====================================================

# @app.get("/health")
# async def health():

#     return {

#         "status": "running",

#         "model": "loaded"

#     }



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

# =========================================================
# APP SETUP
# =========================================================
app = FastAPI(title="Potato Leaf Disease Detection")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = Path(__file__).resolve().parent

# ✅ FIXED SAFE PATHS (IMPORTANT)
STATIC_DIR = BASE_DIR / "static"
TEMPLATE_DIR = BASE_DIR / "templates"

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
templates = Jinja2Templates(directory=str(TEMPLATE_DIR))

# =========================================================
# MODEL LOAD
# =========================================================
MODEL_PATH = BASE_DIR / "model" / "potatoes.keras"
CLASS_NAMES_PATH = BASE_DIR / "class_names.json"

with open(CLASS_NAMES_PATH, "r", encoding="utf-8") as f:
    class_names = json.load(f)

model = tf.keras.models.load_model(MODEL_PATH, compile=False)

print("✅ Model Loaded Successfully")
print("✅ Classes:", class_names)

# =========================================================
# PREDICTION CORE (FIXED)
# =========================================================
def predict_image(image: Image.Image):

    image = ImageOps.exif_transpose(image)
    image = image.convert("RGB")
    image = image.resize((256, 256))

    img_array = tf.keras.preprocessing.image.img_to_array(image)

    # ❌ DO NOT NORMALIZE
    # img_array = img_array / 255.0   ← REMOVE THIS

    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array, verbose=0)

    scores = np.squeeze(predictions)

    idx = int(np.argmax(scores))
    confidence = float(np.max(scores) * 100)

    predicted_class = class_names[idx]

    return predicted_class, round(confidence, 2)


# =========================================================
# STATUS
# =========================================================
def get_status(conf):
    if conf >= 90:
        return "Very Confident"
    elif conf >= 80:
        return "Good Confidence"
    elif conf >= 70:
        return "Medium Confidence"
    return "Low Confidence"


# =========================================================
# CORE PIPELINE (UNCHANGED BUT CLEAN)
# =========================================================
async def run_prediction(file: UploadFile):

    allowed_types = ["image/jpeg", "image/jpg", "image/png", "image/webp"]

    if file.content_type not in allowed_types:
        return {"ok": False, "error": "Only image files allowed"}

    contents = await file.read()

    if not contents:
        return {"ok": False, "error": "Empty file"}

    try:
        image = Image.open(io.BytesIO(contents))

        prediction, confidence = predict_image(image)

        status = get_status(confidence)

        return {
            "ok": True,
            "prediction": prediction,
            "confidence": confidence,
            "status": status
        }

    except Exception as e:
        return {"ok": False, "error": str(e)}


# =========================================================
# ROUTES (YOUR STYLE KEPT)
# =========================================================

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(
        name="index.html",
        request=request,
        context={}
    )


@app.post("/predict", response_class=HTMLResponse)
async def predict_html(request: Request, file: UploadFile = File(...)):

    result = await run_prediction(file)

    return templates.TemplateResponse(
        name="index.html",
        request=request,
        context=result
    )


@app.post("/predict-json")
async def predict_json(file: UploadFile = File(...)):
    return JSONResponse(await run_prediction(file))