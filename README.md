# Anomaly Detection Engine

![Python](https://img.shields.io/badge/python-3.10+-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-009688)
![Vercel](https://img.shields.io/badge/deployed%20on-Vercel-000000)
![License](https://img.shields.io/badge/license-MIT-green)

A dual-model login anomaly detection system combining **Random Forest regression** (supervised heuristic scoring) with **Isolation Forest** (unsupervised outlier detection). Deployed as a FastAPI web service with a single-page animated dashboard.

**Live Demo:** [https://anomaly-detection-eosin.vercel.app](https://anomaly-detection-eosin.vercel.app)

---

## Table of Contents

- [Features](#features)
- [Architecture](#architecture)
- [How It Works](#how-it-works)
- [Model Performance](#model-performance)
- [Quick Start](#quick-start)
- [API Reference](#api-reference)
- [Deployment](#deployment)
- [Project Structure](#project-structure)
- [Limitations](#limitations)

---

## Features

- **Dual-model inference** — supervised score + unsupervised anomaly flag per request
- **Raw value input** — accepts country names, browser names, device types — no manual encoding
- **Animated dashboard UI** — SVG gauge, score distribution chart, staggered entrance animations
- **Detection history** — persisted via MongoDB (optional) with localStorage fallback
- **Feedback loop** — thumbs up/down on past results for future evaluation
- **Graceful degradation** — fully functional without a database
- **Vercel-ready** — serverless deployment with zero configuration

---

## Architecture

```
Login Data (CSV or synthetic)
          |
          v
    pipeline.py        Feature engineering, label encoding, model training
          |
          +-- encoders.pkl          (Country, Browser, Device label maps)
          +-- regressor_model.pkl   (Random Forest regressor)
          +-- isolation_forest.pkl  (Isolation Forest detector)
          |
          v
      app.py            FastAPI server + async MongoDB (motor)
          |
          +-- POST /detect     Run both models on a login event
          +-- GET  /history    Paginated detection records
          +-- DELETE /history   Clear or delete records
          +-- PATCH /history/{id}/feedback  Mark correct/incorrect
          +-- GET  /            Serve web UI
          +-- GET  /health      Model and database status
          |
          v
  templates/index.html  Single-page UI (5 tabs, animated, responsive)
```

---

## How It Works

### Heuristic Target Score

Each login event is scored (0-10) based on behavioral rules derived from known attack patterns:

| Condition | Points |
|-----------|--------|
| Countries used > 2 | +1 |
| Device types > 3 | +1 |
| IP addresses > 4 | +1 |
| Browser categories > 3 | +1 |
| Consecutive logins < 5s apart | +1 |
| Browser identified as bot | +2 |
| Device identified as bot | +2 |
| Login failure ratio > 10 | +1 |

### Model 1 — Random Forest Regressor

Trained to predict the heuristic score from raw login features. While this creates a circular dependency (the model validates the same rules it learns), it provides a smooth, continuous score that is more robust than individual rule checks.

### Model 2 — Isolation Forest

A truly unsupervised detector that identifies outliers by measuring how easily a point can be isolated from the rest. This catches patterns the heuristic rules may miss, offering an independent second opinion.

---

## Model Performance

| Metric | Random Forest |
|--------|--------------|
| MSE | 0.000086 |
| RMSE | 0.009 |
| MAE | 0.0005 |

---

## Quick Start

### Prerequisites

- Python 3.10+
- MongoDB (optional — app runs without it)

### Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Start the server
uvicorn app:app --reload

# Open in browser
open http://localhost:8000
```

### Re-train Models

```bash
# Using synthetic data (default, for demo)
python pipeline.py

# Using real CSV data
python pipeline.py --csv Login_Data.csv
```

---

## API Reference

### Health Check

```
GET /health
```

Returns model load status and MongoDB connection state.

```json
{
  "status": "ok",
  "regressor_loaded": true,
  "isolation_forest_loaded": true,
  "encoders_loaded": true,
  "mongodb_connected": false
}
```

### Detect Anomaly

```
POST /detect
```

Accepts raw login event attributes and returns scores from both models. Results are saved to MongoDB if available.

**Request:**

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
  "id": "665a1b2c...",
  "anomalous_score": 1.01,
  "risk_level": "Low",
  "is_anomalous": false,
  "isolation_forest_anomaly": false
}
```

**cURL Example:**

```bash
curl -X POST https://anomaly-detection-eosin.vercel.app/detect \
  -H "Content-Type: application/json" \
  -d '{"Country":"NO","Device_Type":"mobile","Login_Successful":1,"LoginRatio":0.05,"Browser_Category":"Chrome","Total_Device_Types":2,"Total_IP_Addresses":1,"Total_Countries":1,"Total_Browser_Categories":4,"Time_Difference_in_sec":0.5}'
```

### History

```
GET /history?skip=0&limit=50
```

Returns paginated detection records. When MongoDB is unavailable, the frontend gracefully falls back to browser localStorage.

```
DELETE /history
DELETE /history/{id}
```

Clear all records or delete a specific one.

```
PATCH /history/{id}/feedback
```

Submit feedback on a detection. Accepts `"correct"`, `"incorrect"`, or `null`.

```json
{
  "feedback": "correct"
}
```

---

## Deployment

### Vercel

[![Deploy to Vercel](https://vercel.com/button)](https://vercel.com/import/project?template=https://github.com/sujalthapa369/Anomaly-detection)

1. Push the repository to GitHub
2. Import the project in Vercel (auto-detects `vercel.json`)
3. (Optional) Add `MONGODB_URL` environment variable for persistent history

### MongoDB Atlas (Optional)

For persistent detection history across sessions, create a free MongoDB Atlas cluster and set the `MONGODB_URL` environment variable in your deployment platform:

```env
MONGODB_URL=mongodb+srv://<user>:<password>@<cluster>.mongodb.net/anomaly_detection?retryWrites=true&w=majority
```

The application detects the database connection at startup and degrades gracefully when unavailable.

---

## Project Structure

```
├── app.py                  FastAPI server with all API routes
├── pipeline.py             Training pipeline (encoders + models)
├── db.py                   Async MongoDB client (motor) with offline fallback
├── api/index.py            Vercel serverless entry point
├── templates/index.html    Single-page animated dashboard UI
├── requirements.txt        Python dependencies
├── vercel.json             Vercel deployment configuration
├── render.yaml             Render deployment configuration
├── encoders.pkl            Serialized label encoders
├── regressor_model.pkl     Trained Random Forest regressor
└── isolation_forest.pkl    Trained Isolation Forest detector
```

---

## Limitations

1. **Circular target** — The regressor predicts a score derived from the same features it trains on. It validates heuristic rules rather than discovering novel anomalies. The Isolation Forest partially mitigates this.

2. **Label encoding drift** — Country and browser encoders require periodic retraining as new values appear in production data.

3. **No ground truth** — Without confirmed attack labels, the heuristic score is an approximation. Feedback collection (thumbs up/down) is designed to address this over time.

4. **Local history fallback** — Without MongoDB Atlas, detection history is stored in browser localStorage and does not persist across devices or sessions.

---

## License

MIT
