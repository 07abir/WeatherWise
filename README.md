# WeatherWise — FastAPI + Streamlit

This project uses the five pickle artifacts supplied for the WeatherWise model.

## Detected model configuration

- Estimator: `DecisionTreeClassifier`
- Features: `month`, `day_of_year`, `latitude`, `longitude`
- Classes: `Clear/Sunny`, `Cloudy/Partly Cloudy`, `Rainy/Shower`
- Preprocessing: `SimpleImputer` followed by `StandardScaler`
- Target decoder: `LabelEncoder`

## Important limitation

The saved imputer/scaler show that the model was trained with a single location:
Kolkata (latitude 22.5726, longitude 88.3639). The application therefore only
supports Kolkata. It must be retrained with representative data from multiple
locations before other cities can be supported.

This is a historical-pattern classifier based on calendar date and location.
It is **not** a real weather forecast and cannot know actual future atmospheric
conditions. The displayed class scores are Decision Tree leaf class proportions,
not calibrated probabilities of actual future weather.

## Project structure

```text
WeatherWise/
├── backend/
│   ├── main.py
│   └── models/
│       ├── weather_model.pkl
│       ├── feature_columns.pkl
│       ├── imputer.pkl
│       ├── scaler.pkl
│       └── label_encoder.pkl
├── frontend/
│   └── app.py
└── requirements.txt
```

## Setup on Windows (PowerShell)

Open a terminal in the `WeatherWise` folder.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PowerShell blocks activation, you can instead run:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Run the backend

From the `WeatherWise` project root:

```powershell
uvicorn backend.main:app --reload --host 127.0.0.1 --port 8000
```

API documentation: http://127.0.0.1:8000/docs

Health check: http://127.0.0.1:8000/health

## Run the frontend

Open a **second terminal**, activate the same virtual environment, and run:

```powershell
streamlit run frontend/app.py
```

Streamlit normally opens at http://localhost:8501.

Keep both terminals running while using the application.

## API example

POST `http://127.0.0.1:8000/predict`

```json
{
  "prediction_date": "2026-11-15",
  "location": "Kolkata"
}
```

## Troubleshooting

- **Connection refused:** Start FastAPI before clicking Predict in Streamlit.
- **Missing model artifact:** Ensure all five `.pkl` files remain inside `backend/models/`.
- **Scikit-learn version warning/error:** The supplied artifacts were created with scikit-learn 1.6.1. This project pins that version. Avoid loading pickle files from unknown sources.
- **Other cities rejected:** This is intentional because the current model was trained using Kolkata coordinates only.
