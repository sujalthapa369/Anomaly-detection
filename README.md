# Anomaly Detection Engine

Detect anomalous login behavior using **XGBoost regression** (supervised heuristic) and **Isolation Forest** (unsupervised anomaly detection). Deployed as a FastAPI web service with a modern dashboard UI.

---

## Architecture

```
Login_Data.csv
      |
      v
  pipeline.py          <-- feature engineering, label encoding, model training
      |
      +-- encoders.pkl           (LabelEncoder for Country, Browser, Device)
      +-- regressor_model.pkl    (XGBoost - predicts heuristic anomaly score)
      +-- isolation_forest.pkl   (IsolationForest - unsupervised outlier detection)
      |
      v
    app.py              <-- FastAPI server
      |
      +-- POST /detect  <-- accepts raw values, encodes internally, runs both models
      +-- GET  /        <-- serves web UI
      |
      v
  templates/index.html  <-- Dashboard UI (Detector, Dashboard, History tabs)
```

## How It Works

### The Problem
This project originally created a **rule-based heuristic target** score (0-10):

| Condition | Points |
|-----------|--------|
| Countries used > 2 | +1 |
| Device types > 3 | +1 |
| IP addresses > 4 | +1 |
| Browser categories > 3 | +1 |
| Time diff between 0-5 sec | +1 |
| Browser is a bot | +2 |
| Device type is bot | +2 |
| Login ratio > 10 | +1 |

**XGBoost** was trained to predict this score — which means it learns to approximate its own rules. This is a known limitation (the model validates the heuristic rather than discovering true anomalies).

### The Improvement
**Isolation Forest** is added alongside as a truly unsupervised method. It detects outliers based on how easily a point can be isolated, without requiring any labels.

Both results are returned per request.

## Quick Start

### Local
```bash
pip install -r requirements.txt
uvicorn app:app --reload
# Open http://localhost:8000
```

### Re-train Models (Optional)
```bash
# With synthetic data (for demo):
python pipeline.py

# With real data:
python pipeline.py --csv Login_Data.csv
```

### Docker
```bash
docker build -t anomaly-detector .
docker run -p 8000:8000 anomaly-detector
```

## API Reference

### `POST /detect`

**Request Body:**
```json
{
  "Country": "NO",
  "Device_Type": "mobile",
  "Login_Successful": 1,
  "LoginRatio": 0.05,
  "Browser_Category": "Chrome",
  "Total_Device_Types": 2,
  "Total_IP_Addresses": 1,
  "Total_Countries": 1,
  "Total_Browser_Categories": 4,
  "Time_Difference_in_sec": 0.5
}
```

**Response:**
```json
{
  "anomalous_score": 1.01,
  "risk_level": "Low",
  "is_anomalous": false,
  "isolation_forest_anomaly": false
}
```

**cURL Example:**
```bash
curl -X POST http://localhost:8000/detect \
  -H "Content-Type: application/json" \
  -d '{"Country":"NO","Device_Type":"mobile","Login_Successful":1,"LoginRatio":0.05,"Browser_Category":"Chrome","Total_Device_Types":2,"Total_IP_Addresses":1,"Total_Countries":1,"Total_Browser_Categories":4,"Time_Difference_in_sec":0.5}'
```

### `GET /health`
Returns model load status.

## Deployment

### Render
1. Push to GitHub
2. Create new Web Service
3. Set start command: `uvicorn app:app --host 0.0.0.0 --port $PORT`

### Vercel
1. Push to GitHub
2. Import project in Vercel
3. Auto-detects `vercel.json`

## Dataset

- **31.2 million** login records
- **4.3 million** unique users
- **217** countries, **5** device types, ~600 browser categories
- Time range: Feb 2020 - Feb 2021
- A single outlier user had **14M+ failed attempts** (bot)

## Model Performance

| Metric | XGBoost |
|--------|---------|
| MSE | 0.003 |
| RMSE | 0.055 |
| MAE | 0.007 |

## Limitations

1. **Circular target**: XGBoost predicts a score derived from the same features it trains on — it validates heuristic rules rather than detecting novel anomalies.
2. **Label encoding**: Country and browser encoders need periodic retraining as new values appear.
3. **No ground truth**: There are no confirmed "attack" labels; the heuristic is an approximation.

## Project Structure

```
├── app.py                # FastAPI backend
├── pipeline.py           # Training pipeline (encoders + models)
├── requirements.txt
├── Dockerfile
├── vercel.json
├── render.yaml
├── encoders.pkl          # Label encoders for raw value mapping
├── regressor_model.pkl   # Trained XGBoost
├── isolation_forest.pkl  # Trained IsolationForest
├── api/index.py          # Vercel entry point
└── templates/index.html  # Web UI
```
