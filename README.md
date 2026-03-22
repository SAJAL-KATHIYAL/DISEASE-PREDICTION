# Disease Prediction System

A Flask-based web application for disease prediction using machine learning.

## Features

- 🎯 Predict disease presence based on patient health metrics
- 📊 Display confidence scores and probabilities
- 🎨 Clean, responsive web interface
- 🔒 Input validation and error handling
- 📱 Mobile-friendly design

## Health Metrics

The system predicts disease based on:
- **Age**: Patient age in years
- **Blood Pressure**: Systolic/diastolic in mmHg
- **Cholesterol**: Total cholesterol in mg/dL
- **Blood Sugar**: Fasting blood glucose in mg/dL
- **BMI**: Body Mass Index in kg/m²
- **Heart Rate**: Resting heart rate in bpm

## Installation

1. Create and activate virtual environment:
```bash
python -m venv venv
.\venv\Scripts\Activate.ps1  # Windows PowerShell
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Train the model:
```bash
python train.py
```

This will create `model.pkl` and `scaler.pkl` files.

## Running the App

```bash
python app.py
```

The application will be available at `http://localhost:5000`

## Project Structure

```
disease_prediction/
├── app.py                  # Flask application
├── train.py               # Model training script
├── requirements.txt       # Python dependencies
├── model.pkl             # Trained model (generated)
├── scaler.pkl            # Feature scaler (generated)
├── templates/
│   └── index.html        # Web interface
└── README.md            # This file
```

## API Endpoints

### GET `/`
Home page with prediction form

### POST `/predict`
Make a prediction with patient metrics
- **Request**: JSON with features
- **Response**: Prediction result with confidence

### GET `/api/features`
Get list of available features

## Example Request

```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "Age": 45,
    "Blood Pressure": 130,
    "Cholesterol": 240,
    "Blood Sugar": 110,
    "BMI": 26.5,
    "Heart Rate": 75
  }'
```

## Example Response

```json
{
  "prediction": 1,
  "disease_present": true,
  "confidence": 0.85,
  "probability_healthy": 0.15,
  "probability_disease": 0.85
}
```

## Model Details

- **Algorithm**: Random Forest Classifier
- **Features**: 6 health metrics
- **Training Samples**: 200
- **Preprocessing**: StandardScaler normalization

## Note

This is a demonstration system. For actual medical use, consult with healthcare professionals and use validated models with proper clinical testing.
