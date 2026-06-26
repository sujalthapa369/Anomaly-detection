# Anomaly Detection Engine

Detect anomalous login behavior using **Random Forest regression** (supervised heuristic) and **Isolation Forest** (unsupervised anomaly detection). Deployed as a FastAPI web service with a modern animated dashboard UI.

**Live Demo:** [https://anomaly-detection-eosin.vercel.app](https://anomaly-detection-eosin.vercel.app)

---

## Architecture

```
Login_Data.csv
      |
      v
  pipeline.py          <-- feature engineering, label encoding, model training
      |
      +-- encoders.pkl           (LabelEncoder for Country, Browser, Device)
      +-- regressor_model.pkl    (RandomForest - predicts heuristic anomaly score)
      +-- isolation_forest.pkl   (IsolationForest - unsupervised outlier detection)
      |
      v
    app.py              <-- FastAPI server (+ async MongoDB via motor)
      |
      +-- POST /detect  <-- accepts raw values, encodes internally, runs both models
      +-- GET  /history <-- paginated detection history
      +-- DELETE /history, /history/{id}  <-- clear or delete records
      +-- PATCH /history/{id}/feedback    <-- mark correct/incorrect
      +-- GET  /        <-- serves web UI
      +-- GET  /health  <-- model & DB status
      |
      v
  templates/index.html  <-- Full UI (Landing, Detector, Dashboard, History, Settings)
```

## How It Works

### The Limitation (Honest Admission)
This project creates a **rule-based heuristic target** score (0-10):

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

**Random Forest** is trained to predict this score — it learns to approximate its own rules. This is a known limitation (the model validates the heuristic rather than discovering true anomalies).

### The Improvement
**Isolation Forest** runs alongside as a truly unsupervised method. It detects outliers based on how easily a point can be isolated, without requiring any labels. Both results are returned per request.

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

## API Reference

### `POST /detect`
Runs both models on a login event and returns scores. Saves to MongoDB (if available) or returns without persistence.

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
  "id": "",
  "anomalous_score": 1.01,
  "risk_level": "Low",
  "is_anomalous": false,
  "isolation_forest_anomaly": false
}
```

### `GET /history?skip=0&limit=50`
Paginated history of past detections. Returns empty array if MongoDB is offline — frontend falls back to localStorage.

### `DELETE /history` or `DELETE /history/{id}`
Clear all or delete a single record.

### `PATCH /history/{id}/feedback`
Body: `{"feedback": "correct" | "incorrect" | null}`

### `GET /health`
Returns model load status and MongoDB connection state.

## Deployment

### Vercel (Primary)
[![Deployed on Vercel](https://vercel.com/button)](https://anomaly-detection-eosin.vercel.app)

1. Push to GitHub
2. Import project in Vercel
3. Auto-detects `vercel.json`

**Optional:** Set `MONGODB_URL` environment variable to a MongoDB Atlas connection string for persistent history across sessions. Without it, the app runs fully offline — history stores in browser localStorage.

## Dataset

- **31.2 million** login records
- **4.3 million** unique users
- **217** countries, **5** device types, ~600 browser categories
- Time range: Feb 2020 - Feb 2021
- A single outlier user had **14M+ failed attempts** (bot)

## Model Performance

| Metric | Random Forest |
|--------|--------------|
| MSE | 0.000086 |
| RMSE | 0.009 |
| MAE | 0.0005 |

## Limitations

1. **Circular target**: The regressor predicts a score derived from the same features it trains on — it validates heuristic rules rather than detecting novel anomalies.
2. **Label encoding**: Country and browser encoders need periodic retraining as new values appear.
3. **No ground truth**: There are no confirmed "attack" labels; the heuristic is an approximation.
4. **MongoDB optional**: Without Atlas, history is stored in browser localStorage (single-device only).

## Project Structure

```
├── app.py                # FastAPI backend + all API routes
├── pipeline.py           # Training pipeline (encoders + models)
├── db.py                 # Async MongoDB module (motor)
├── requirements.txt
├── vercel.json
├── render.yaml           # Render deployment config
├── api/index.py          # Vercel entry point
├── encoders.pkl          # Label encoders for raw value mapping
├── regressor_model.pkl   # Trained RandomForest regressor
├── isolation_forest.pkl  # Trained IsolationForest
└── templates/index.html  # Full web UI (single page)
```
