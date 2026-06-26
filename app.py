import os, pickle, re
import numpy as np
from pathlib import Path
from datetime import datetime, timezone
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
import uvicorn
import db as database

ROOT = Path(__file__).parent

# ---------- lifecycle ----------
@asynccontextmanager
async def lifespan(app: FastAPI):
    await database.connect_db()
    yield
    await database.close_db()

app = FastAPI(title="Anomaly Detection API", version="3.0.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"])

# ---------- load artifacts ----------
def load_artifact(name):
    p = ROOT / name
    if p.exists():
        with open(p, "rb") as f:
            return pickle.load(f)
    return None

model_reg = load_artifact("regressor_model.pkl")
model_ifo = load_artifact("isolation_forest.pkl")
encoders  = load_artifact("encoders.pkl")

# ---------- preprocessing helpers ----------
DEVICE_MAP = {"mobile": 0, "desktop": 1, "tablet": 2, "bot": 3, "unknown": 4}

BOT_KW = ["bot", "awariosmartbot", "metajobbot", "libwwwperl",
    "mobileiron", "coc coc", "woobot", "crawler_faq", "job roboter",
    "keeper", "bingpreview", "nutch", "curl", "okhttp", "zipppbot"]

TOP_BROWSERS = [
    "Chrome", "Chrome Mobile", "Chrome Mobile WebView", "Firefox",
    "Firefox Mobile", "Safari", "Safari Mobile", "Android", "Edge",
    "MiuiBrowser", "Opera", "Opera Mobile", "Samsung Internet",
    "Facebook", "Instagram", "Yandex Browser", "Maxthon", "UC Browser",
    "Vivaldi", "Brave", "QQbrowser", "Sogou Explorer"
]

def categorize_browser(name: str) -> str:
    m = re.search(r"^\D+", name)
    clean = m.group() if m else name
    if clean in TOP_BROWSERS:
        return clean
    if any(k in clean.lower() for k in BOT_KW):
        return "Bot"
    return "Others"

def encode_country(val: str) -> int:
    if encoders and "country" in encoders:
        try:
            return int(encoders["country"].transform([str(val)])[0])
        except:
            pass
    try:
        return int(val)
    except:
        return 0

def encode_browser(val: str) -> int:
    cat = categorize_browser(val)
    if encoders and "browser" in encoders:
        try:
            return int(encoders["browser"].transform([cat])[0])
        except:
            pass
    try:
        return int(val)
    except:
        return 0

def encode_device(val: str) -> int:
    if isinstance(val, str) and val.lower() in DEVICE_MAP:
        return DEVICE_MAP[val.lower()]
    try:
        return int(val)
    except:
        return 0

# ---------- models ----------
class AnomalyInput(BaseModel):
    Country: str = Field(..., description="Country name or encoded int")
    Device_Type: str = Field(..., description="Device type (mobile/desktop/tablet/bot/unknown) or encoded int")
    Login_Successful: int = Field(..., ge=0, le=1)
    LoginRatio: float = Field(..., ge=0)
    Browser_Category: str = Field(..., description="Browser name or encoded int")
    Total_Device_Types: int = Field(..., ge=0)
    Total_IP_Addresses: int = Field(..., ge=0)
    Total_Countries: int = Field(..., ge=0)
    Total_Browser_Categories: int = Field(..., ge=0)
    Time_Difference_in_sec: float = Field(..., ge=0)

class AnomalyResponse(BaseModel):
    id: str = ""
    anomalous_score: float
    risk_level: str
    is_anomalous: bool
    isolation_forest_anomaly: bool = False

class HistoryRecord(BaseModel):
    id: str
    created_at: str
    anomalous_score: float
    risk_level: str
    is_anomalous: bool
    isolation_forest_anomaly: bool
    feedback: str | None = None

class HistoryListResponse(BaseModel):
    records: list[HistoryRecord]
    total: int

# ---------- routes ----------
@app.get("/", response_class=HTMLResponse)
async def index():
    p = ROOT / "templates" / "index.html"
    if p.exists():
        return HTMLResponse(p.read_text(encoding="utf-8"))
    return HTMLResponse("<h1>Anomaly Detection API</h1><p>POST /detect</p>")

@app.post("/detect", response_model=AnomalyResponse)
async def detect(data: AnomalyInput):
    if model_reg is None:
        raise HTTPException(500, "Regression model not loaded")

    country_enc = encode_country(data.Country)
    device_enc  = encode_device(data.Device_Type)
    browser_enc = encode_browser(data.Browser_Category)

    features = np.array([[country_enc, device_enc, data.Login_Successful,
        data.LoginRatio, browser_enc, data.Total_Device_Types,
        data.Total_IP_Addresses, data.Total_Countries,
        data.Total_Browser_Categories, data.Time_Difference_in_sec
    ]])

    score = float(model_reg.predict(features)[0])
    score = round(max(0, min(10, score)), 2)

    if score >= 3:
        risk, is_anom = "High", True
    elif score >= 1.5:
        risk, is_anom = "Medium", False
    else:
        risk, is_anom = "Low", False

    ifo_anom = False
    if model_ifo is not None:
        ifo_pred = model_ifo.predict(features)[0]
        ifo_anom = bool(ifo_pred == -1)

    record_id = await database.save_detection({
        "country": data.Country,
        "device_type": data.Device_Type,
        "login_successful": data.Login_Successful,
        "login_ratio": data.LoginRatio,
        "browser_category": data.Browser_Category,
        "total_device_types": data.Total_Device_Types,
        "total_ip_addresses": data.Total_IP_Addresses,
        "total_countries": data.Total_Countries,
        "total_browser_categories": data.Total_Browser_Categories,
        "time_difference_sec": data.Time_Difference_in_sec,
        "anomalous_score": score,
        "risk_level": risk,
        "is_anomalous": is_anom,
        "isolation_forest_anomaly": ifo_anom,
    })

    return AnomalyResponse(
        id=record_id or "",
        anomalous_score=score, risk_level=risk,
        is_anomalous=is_anom, isolation_forest_anomaly=ifo_anom
    )

@app.get("/history", response_model=HistoryListResponse)
async def history(skip: int = Query(0, ge=0), limit: int = Query(50, ge=1, le=200)):
    records = await database.get_history(skip=skip, limit=limit)
    total = await database.count_detections()
    return HistoryListResponse(records=records, total=total)

@app.delete("/history")
async def clear_history(ids: str = Query("")):
    if ids:
        await database.delete_history([i.strip() for i in ids.split(",") if i.strip()])
    else:
        await database.delete_history()
    return {"ok": True}

@app.delete("/history/{id}")
async def delete_record(id: str):
    ok = await database.delete_one_detection(id)
    if not ok:
        raise HTTPException(404, "Record not found")
    return {"ok": True}

@app.patch("/history/{id}/feedback")
async def set_feedback(id: str, body: dict):
    fb = body.get("feedback")
    if fb not in ("correct", "incorrect", None):
        raise HTTPException(400, "feedback must be 'correct', 'incorrect', or null")
    ok = await database.update_feedback(id, fb)
    if not ok:
        raise HTTPException(404, "Record not found")
    return {"ok": True}

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "regressor_loaded": model_reg is not None,
        "isolation_forest_loaded": model_ifo is not None,
        "encoders_loaded": encoders is not None,
        "mongodb_connected": database.available
    }

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("app:app", host="0.0.0.0", port=port)
