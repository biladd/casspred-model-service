from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional
import joblib
import json
import numpy as np
import pandas as pd
from datetime import datetime
import os

# ── Load model & metadata ──
BASE_DIR = os.path.dirname(__file__)
MODEL_DIR = os.path.join(BASE_DIR, "models")

model_A  = joblib.load(os.path.join(MODEL_DIR, "final_model_taskA.pkl"))
model_B  = joblib.load(os.path.join(MODEL_DIR, "final_model_taskB.pkl"))
scaler   = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))

with open(os.path.join(MODEL_DIR, "final_metadata.json")) as f:
    metadata = json.load(f)

FEATURES       = metadata["features"]          # 83 fitur
THRESHOLD_DEF  = metadata["task_A"]["threshold_default"]       # 0.20
THRESHOLD_R80  = metadata["task_A"]["threshold_recall80"]      # 0.22
HEALTH_LABELS  = metadata["task_B"]["labels"]  # {0: Healthy, 1: Warning, 2: Critical}

# ── App ──
app = FastAPI(
    title="Predictive Maintenance API",
    description="API untuk prediksi kegagalan mesin industri — Proyek LapisAI × PNJ",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # ganti dengan domain Next.js saat production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Schema Input ──
class SensorInput(BaseModel):
    # Raw sensor (wajib diisi)
    machine_id: str = Field(..., example="M-01")
    temperature: float = Field(..., example=75.5)
    vibration: float = Field(..., ge=0, example=0.6)
    pressure: float = Field(..., example=102.3)
    rpm: int = Field(..., example=2400)
    power_consumption: float = Field(..., example=78.0)
    noise_level: float = Field(..., example=71.0)
    humidity: float = Field(..., example=55.0)
    operating_hours: float = Field(..., example=1200.0)

    # Fitur rolling (opsional — kalau tidak ada, pakai nilai sensor saat ini)
    temperature_roll_mean_24h: Optional[float] = None
    temperature_roll_std_24h:  Optional[float] = None
    temperature_roll_max_24h:  Optional[float] = None
    temperature_roll_min_24h:  Optional[float] = None
    vibration_roll_std_24h:    Optional[float] = None
    vibration_roll_min_24h:    Optional[float] = None
    pressure_roll_std_24h:     Optional[float] = None
    pressure_roll_max_24h:     Optional[float] = None
    pressure_roll_min_24h:     Optional[float] = None
    rpm_roll_mean_24h:         Optional[float] = None
    rpm_roll_std_24h:          Optional[float] = None
    rpm_roll_max_24h:          Optional[float] = None
    rpm_roll_min_24h:          Optional[float] = None
    power_consumption_roll_mean_24h: Optional[float] = None
    power_consumption_roll_std_24h:  Optional[float] = None
    power_consumption_roll_max_24h:  Optional[float] = None
    power_consumption_roll_min_24h:  Optional[float] = None
    noise_level_roll_std_24h:  Optional[float] = None
    noise_level_roll_max_24h:  Optional[float] = None
    noise_level_roll_min_24h:  Optional[float] = None
    humidity_roll_mean_24h:    Optional[float] = None
    humidity_roll_std_24h:     Optional[float] = None
    humidity_roll_max_24h:     Optional[float] = None
    humidity_roll_min_24h:     Optional[float] = None
    operating_hours_roll_std_24h: Optional[float] = None

    # Fitur delta (opsional)
    temperature_delta:      Optional[float] = None
    vibration_delta:        Optional[float] = None
    pressure_delta:         Optional[float] = None
    rpm_delta:              Optional[float] = None
    power_consumption_delta:Optional[float] = None
    noise_level_delta:      Optional[float] = None
    humidity_delta:         Optional[float] = None
    operating_hours_delta:  Optional[float] = None

    # Fitur lag (opsional)
    temperature_lag_1h:  Optional[float] = None
    temperature_lag_2h:  Optional[float] = None
    temperature_lag_3h:  Optional[float] = None
    temperature_lag_6h:  Optional[float] = None
    temperature_lag_12h: Optional[float] = None
    temperature_lag_24h: Optional[float] = None
    vibration_lag_1h:    Optional[float] = None
    vibration_lag_2h:    Optional[float] = None
    vibration_lag_3h:    Optional[float] = None
    vibration_lag_6h:    Optional[float] = None
    vibration_lag_12h:   Optional[float] = None
    vibration_lag_24h:   Optional[float] = None
    pressure_lag_1h:     Optional[float] = None
    pressure_lag_2h:     Optional[float] = None
    pressure_lag_3h:     Optional[float] = None
    pressure_lag_6h:     Optional[float] = None
    pressure_lag_12h:    Optional[float] = None
    pressure_lag_24h:    Optional[float] = None
    rpm_lag_1h:          Optional[float] = None
    rpm_lag_2h:          Optional[float] = None
    rpm_lag_3h:          Optional[float] = None
    rpm_lag_6h:          Optional[float] = None
    rpm_lag_12h:         Optional[float] = None
    rpm_lag_24h:         Optional[float] = None

    # Fitur interaksi (opsional)
    pressure_x_rpm:      Optional[float] = None
    temp_x_pressure:     Optional[float] = None
    power_per_rpm:       Optional[float] = None
    fft_vib_dominant_freq: Optional[float] = None

    # Time features (opsional — auto-generate dari waktu sekarang)
    hour_of_day:  Optional[int] = None
    day_of_week:  Optional[int] = None
    day_of_month: Optional[int] = None
    month:        Optional[int] = None
    is_weekend:   Optional[int] = None
    is_night:     Optional[int] = None

    # NLP / maintenance features (opsional)
    last_maintenance_type:             Optional[float] = None
    last_maintenance_severity_score:   Optional[float] = None
    last_problem_category:             Optional[float] = None
    last_has_urgent_flag:              Optional[float] = None
    last_technical_term_count:         Optional[float] = None
    last_problem_keyword_count:        Optional[float] = None
    last_action_keyword_count:         Optional[float] = None
    days_since_last_maintenance:       Optional[float] = None

    # Threshold (opsional — default 0.20)
    threshold: Optional[float] = Field(None, ge=0.0, le=1.0, example=0.20)


def build_feature_vector(data: SensorInput) -> np.ndarray:
    """Bangun vektor 83 fitur dari input sensor."""
    now = datetime.now()
    d   = data.dict()

    # Auto-fill time features
    d["hour_of_day"]  = d["hour_of_day"]  if d["hour_of_day"]  is not None else now.hour
    d["day_of_week"]  = d["day_of_week"]  if d["day_of_week"]  is not None else now.weekday()
    d["day_of_month"] = d["day_of_month"] if d["day_of_month"] is not None else now.day
    d["month"]        = d["month"]        if d["month"]        is not None else now.month
    d["is_weekend"]   = d["is_weekend"]   if d["is_weekend"]   is not None else int(now.weekday() >= 5)
    d["is_night"]     = d["is_night"]     if d["is_night"]     is not None else int(now.hour >= 22 or now.hour <= 5)

    # Auto-fill interaction features
    d["pressure_x_rpm"]  = d["pressure_x_rpm"]  or (d["pressure"] * d["rpm"])
    d["temp_x_pressure"] = d["temp_x_pressure"] or (d["temperature"] * d["pressure"])
    d["power_per_rpm"]   = d["power_per_rpm"]   or (d["power_consumption"] / (d["rpm"] + 1))
    d["fft_vib_dominant_freq"] = d["fft_vib_dominant_freq"] or 0.0

    # Auto-fill rolling dengan nilai sensor saat ini (fallback)
    rolling_defaults = {
        "temperature_roll_mean_24h": d["temperature"],
        "temperature_roll_std_24h":  0.0,
        "temperature_roll_max_24h":  d["temperature"],
        "temperature_roll_min_24h":  d["temperature"],
        "vibration_roll_std_24h":    0.0,
        "vibration_roll_min_24h":    d["vibration"],
        "pressure_roll_std_24h":     0.0,
        "pressure_roll_max_24h":     d["pressure"],
        "pressure_roll_min_24h":     d["pressure"],
        "rpm_roll_mean_24h":         d["rpm"],
        "rpm_roll_std_24h":          0.0,
        "rpm_roll_max_24h":          d["rpm"],
        "rpm_roll_min_24h":          d["rpm"],
        "power_consumption_roll_mean_24h": d["power_consumption"],
        "power_consumption_roll_std_24h":  0.0,
        "power_consumption_roll_max_24h":  d["power_consumption"],
        "power_consumption_roll_min_24h":  d["power_consumption"],
        "noise_level_roll_std_24h":  0.0,
        "noise_level_roll_max_24h":  d["noise_level"],
        "noise_level_roll_min_24h":  d["noise_level"],
        "humidity_roll_mean_24h":    d["humidity"],
        "humidity_roll_std_24h":     0.0,
        "humidity_roll_max_24h":     d["humidity"],
        "humidity_roll_min_24h":     d["humidity"],
        "operating_hours_roll_std_24h": 0.0,
    }
    for k, v in rolling_defaults.items():
        if d.get(k) is None:
            d[k] = v

    # Auto-fill delta & lag dengan 0 (tidak ada history)
    for feat in FEATURES:
        if d.get(feat) is None:
            d[feat] = 0.0

    # NLP defaults (belum ada maintenance = -1)
    nlp_defaults = {
        "last_maintenance_type": -1,
        "last_maintenance_severity_score": -1,
        "last_problem_category": -1,
        "last_has_urgent_flag": -1,
        "last_technical_term_count": -1,
        "last_problem_keyword_count": -1,
        "last_action_keyword_count": -1,
        "days_since_last_maintenance": -1,
    }
    for k, v in nlp_defaults.items():
        if d.get(k) in [None, 0.0]:
            d[k] = v

    # Susun sesuai urutan FEATURES
    vec = np.array([d[f] for f in FEATURES], dtype=float).reshape(1, -1)
    return vec


# ── Endpoints ──

@app.get("/")
def root():
    return {
        "name": "Predictive Maintenance API",
        "version": "1.0.0",
        "endpoints": {
            "POST /predict"       : "Prediksi lengkap (Task A + Task B)",
            "POST /predict/failure": "Hanya prediksi kegagalan 7 hari (Task A)",
            "POST /predict/health" : "Hanya prediksi status kesehatan (Task B)",
            "GET  /health"        : "Cek API status",
            "GET  /model/info"    : "Info model & threshold",
        }
    }


@app.get("/health")
def health_check():
    return {"status": "ok", "timestamp": datetime.now().isoformat()}


@app.get("/model/info")
def model_info():
    return {
        "task_A": {
            "description"   : "Prediksi apakah mesin akan gagal dalam 7 hari",
            "model"         : metadata["task_A"]["model"],
            "roc_auc"       : metadata["task_A"]["roc_auc"],
            "pr_auc"        : metadata["task_A"]["pr_auc"],
            "threshold_default"  : THRESHOLD_DEF,
            "threshold_recall80" : THRESHOLD_R80,
            "threshold_options"  : [0.08, 0.15, 0.20, 0.30, 0.40, 0.50],
        },
        "task_B": {
            "description": "Status kesehatan mesin (Healthy / Warning / Critical)",
            "model"      : metadata["task_B"]["model"],
            "macro_f1"   : metadata["task_B"]["macro_f1"],
            "labels"     : HEALTH_LABELS,
        },
        "n_features": metadata["n_features"],
    }


@app.post("/predict")
def predict_all(data: SensorInput):
    """Prediksi lengkap: Task A (failure) + Task B (health status)."""
    try:
        vec       = build_feature_vector(data)
        vec_scaled = scaler.transform(vec)
        threshold  = data.threshold if data.threshold is not None else THRESHOLD_DEF

        # Task A — failure prediction
        proba_A      = float(model_A.predict_proba(vec_scaled)[0][1])
        will_fail    = bool(proba_A >= threshold)
        risk_level   = (
            "HIGH"   if proba_A >= 0.50 else
            "MEDIUM" if proba_A >= 0.20 else
            "LOW"
        )

        # Task B — health status
        pred_B       = int(model_B.predict(vec_scaled)[0])
        proba_B      = model_B.predict_proba(vec_scaled)[0].tolist()
        health_label = HEALTH_LABELS[str(pred_B)]

        return {
            "machine_id"  : data.machine_id,
            "timestamp"   : datetime.now().isoformat(),
            "task_A": {
                "will_fail_within_7days": will_fail,
                "failure_probability"   : round(proba_A, 4),
                "risk_level"            : risk_level,
                "threshold_used"        : threshold,
            },
            "task_B": {
                "health_status"   : pred_B,
                "health_label"    : health_label,
                "probabilities"   : {
                    "Healthy" : round(proba_B[0], 4),
                    "Warning" : round(proba_B[1], 4),
                    "Critical": round(proba_B[2], 4),
                },
            },
            "recommendation": _get_recommendation(will_fail, health_label, proba_A),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/failure")
def predict_failure(data: SensorInput):
    """Hanya Task A: prediksi kegagalan dalam 7 hari."""
    try:
        vec        = build_feature_vector(data)
        vec_scaled = scaler.transform(vec)
        threshold  = data.threshold if data.threshold is not None else THRESHOLD_DEF
        proba_A    = float(model_A.predict_proba(vec_scaled)[0][1])
        will_fail  = bool(proba_A >= threshold)

        return {
            "machine_id"             : data.machine_id,
            "timestamp"              : datetime.now().isoformat(),
            "will_fail_within_7days" : will_fail,
            "failure_probability"    : round(proba_A, 4),
            "risk_level"             : "HIGH" if proba_A>=0.50 else "MEDIUM" if proba_A>=0.20 else "LOW",
            "threshold_used"         : threshold,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/health")
def predict_health(data: SensorInput):
    """Hanya Task B: status kesehatan mesin."""
    try:
        vec        = build_feature_vector(data)
        vec_scaled = scaler.transform(vec)
        pred_B     = int(model_B.predict(vec_scaled)[0])
        proba_B    = model_B.predict_proba(vec_scaled)[0].tolist()

        return {
            "machine_id"  : data.machine_id,
            "timestamp"   : datetime.now().isoformat(),
            "health_status": pred_B,
            "health_label" : HEALTH_LABELS[str(pred_B)],
            "probabilities": {
                "Healthy" : round(proba_B[0], 4),
                "Warning" : round(proba_B[1], 4),
                "Critical": round(proba_B[2], 4),
            },
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def _get_recommendation(will_fail: bool, health_label: str, proba: float) -> str:
    if will_fail and health_label == "Critical":
        return "⛔ SEGERA lakukan maintenance darurat! Risiko kegagalan sangat tinggi."
    elif will_fail:
        return "⚠️ Jadwalkan maintenance dalam 7 hari ke depan."
    elif health_label == "Warning":
        return "🔔 Pantau mesin lebih ketat. Pertimbangkan maintenance preventif."
    else:
        return "✅ Mesin dalam kondisi normal. Lanjutkan jadwal maintenance rutin."
