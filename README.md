# Potato Leaf Disease Detection

A FastAPI web application that predicts potato leaf disease from uploaded images using a trained TensorFlow CNN model.

## Features

- Upload potato leaf images from the browser
- Predict one of these classes:
  - Early_Blight
  - Healthy
  - Late_Blight
- Show confidence percentage and confidence status:
  - Very Confident (>= 90)
  - Good Confidence (>= 80)
  - Acceptable Confidence (>= 70)
  - Low Confidence - Please upload a clearer potato leaf image (< 70)
- Modern frontend with image preview and result card
- JSON prediction endpoint for frontend/API use

## Project Structure

- `main.py` - FastAPI backend and prediction logic
- `templates/index.html` - frontend page
- `static/style.css` - frontend styles
- `model/potato_disease_model.keras` - trained model (preferred)
- `model/potato_disease_model.h5` - trained model fallback
- `class_names.json` - model class labels
- `requirements.txt` - Python dependencies

## Requirements

- Python 3.10-3.12 recommended
- Windows users: Microsoft Visual C++ Redistributable (for TensorFlow)

## Installation

1. Open terminal in project folder.
2. Create virtual environment:

```powershell
python -m venv venv
```

3. Activate environment:

```powershell
.\venv\Scripts\activate
```

4. Install dependencies:

```powershell
pip install -r requirements.txt
```

## Run the App

```powershell
python -m uvicorn app:app --reload
```

Then open:

- http://127.0.0.1:8000

## API Endpoints

- `GET /` - Web app UI
- `POST /predict-json` - Predict from uploaded image (multipart/form-data, field name: `file`)
- `POST /predict` - Template response prediction route

## Example Prediction Response

```json
{
  "ok": true,
  "prediction": "Late_Blight",
  "confidence": 70.0,
  "status": "Acceptable Confidence",
  "error": null,
  "uploaded_filename": "leaf.jpg",
  "model_name": "potato_disease_model.keras"
}
```

## Common Issues

### 1) TensorFlow DLL error on Windows

If you see error about `msvcp140_1.dll`, install Microsoft Visual C++ Redistributable (x64), then restart terminal/VS Code.

### 2) Port already in use

If port 8000 is busy, run on another port:

```powershell
python -m uvicorn main:app --reload --port 8011
```

### 3) Frontend shows connection error

- Ensure FastAPI server is running
- Open the app from FastAPI URL (not raw file preview)
- Hard refresh browser (Ctrl+F5)

## Notes

- Model loading is lazy (loaded on first prediction request).
- Backend includes CORS support for local frontend testing.
