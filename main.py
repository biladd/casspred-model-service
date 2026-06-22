from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List
import joblib
import json
import os
from datetime import datetime
from contextlib import asynccontextmanager

import tensorflow as tf
from tensorflow import keras  # pylint: disable=no-name-in-module
import numpy as np
import pandas as pd
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env.local"))


# ────────────────────────────────────────────────────────────────
# PATHS & CONFIG
# ────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(__file__)
MODEL_DIR = os.path.join(BASE_DIR, "models")

SEQ_LENGTH = 24            # LSTM window
RUL_CAP = 168.0            # 7 hari

GROQ_API_URL = os.getenv("GROQ_API_URL")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
# ────────────────────────────────────────────────────────────────
# LOAD MODELS (sekali saat import)
# ────────────────────────────────────────────────────────────────
print("🔄 Loading models...")

# ── Task A & B — Random Forest ──
model_A = joblib.load(os.path.join(MODEL_DIR, "tahap4_taskA_rf_model.pkl"))
model_B = joblib.load(os.path.join(MODEL_DIR, "tahap4_taskB_rf_model.pkl"))
scaler = joblib.load(os.path.join(MODEL_DIR, "tahap4_scaler.pkl"))

with open(os.path.join(MODEL_DIR, "final_metadata.json")) as f:
    metadata = json.load(f)

# Gunakan feature names dari scaler (ground truth) — 84 fitur
FEATURES = list(scaler.feature_names_in_)
print(f"   📋 Using {len(FEATURES)} features from scaler (overrides metadata)")
THRESHOLD_DEF = metadata["task_A"]["threshold_default"]    # 0.20
THRESHOLD_R80 = metadata["task_A"]["threshold_recall80"]   # 0.23
HEALTH_LABELS = metadata["task_B"]["labels"]               # {0:Healthy, 1:Warning, 2:Critical}
print(f"   ✓ Task A & B (RF) loaded — {len(FEATURES)} features")

# ── Task C — LSTM RUL ──
lstm_rul = keras.models.load_model(  # ⭐ Ganti tf.keras → keras
    os.path.join(MODEL_DIR, "tahap5_lstm_rul_model.keras"), compile=False
)
scaler_rul_X = joblib.load(os.path.join(MODEL_DIR, "tahap5_rul_scaler_X.pkl"))
scaler_rul_y = joblib.load(os.path.join(MODEL_DIR, "tahap5_rul_scaler_y.pkl"))
with open(os.path.join(MODEL_DIR, "tahap5_bias_correction.json")) as f:
    bias_data = json.load(f)
RUL_BIAS = bias_data["bias_correction_hours"]
LSTM_N_FEATURES = scaler_rul_X.n_features_in_
print(f"   ✓ LSTM RUL loaded (n_features={LSTM_N_FEATURES}, bias={RUL_BIAS:.2f}h)")

# ── Tahap 6 — Isolation Forest ──
iforest = joblib.load(os.path.join(MODEL_DIR, "tahap6_iforest_model.pkl"))

# tahap6_scaler.pkl berisi dict {'imputer': SimpleImputer, 'scaler': StandardScaler}
iforest_preprocess = joblib.load(os.path.join(MODEL_DIR, "tahap6_scaler.pkl"))
iforest_imputer = iforest_preprocess["imputer"]
scaler_iforest = iforest_preprocess["scaler"]

iforest_threshold_raw = joblib.load(os.path.join(MODEL_DIR, "tahap6_threshold.pkl"))

# Robust threshold parser
if isinstance(iforest_threshold_raw, dict):
    IFOREST_THRESHOLD = float(
        iforest_threshold_raw.get("threshold")
        or iforest_threshold_raw.get("best_threshold")
        or iforest_threshold_raw.get("optimal")
        or list(iforest_threshold_raw.values())[0]
    )
elif isinstance(iforest_threshold_raw, (list, tuple, np.ndarray)):
    IFOREST_THRESHOLD = float(iforest_threshold_raw[0])
elif np.isscalar(iforest_threshold_raw):
    IFOREST_THRESHOLD = float(iforest_threshold_raw)
else:
    with open(os.path.join(MODEL_DIR, "tahap6_metrics.json")) as f:
        IFOREST_THRESHOLD = float(json.load(f)["threshold"])

IFOREST_N_FEATURES = scaler_iforest.n_features_in_
print(f"   ✓ IsolationForest loaded (n_features={IFOREST_N_FEATURES}, threshold={IFOREST_THRESHOLD:.4f})")

# ⭐ PRODUCTION ADJUSTMENT
# Training threshold (0.506) menghasilkan score ~0.6 di production karena
# distribution shift (single-row inference vs 24h window training).
# Threshold baru: P95 training distribution = 0.62 → ~5% flagged.
IFOREST_THRESHOLD_PROD = 0.62
print(f"   ⚙ Production threshold override: {IFOREST_THRESHOLD_PROD:.4f}")
print(f"   📝 Reason: Distribution shift (training mean ~0.44, production ~0.60)")
IFOREST_THRESHOLD = IFOREST_THRESHOLD_PROD

# ────────────────────────────────────────────────────────────────
# APP
# ────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Predictive Maintenance API",
    description="API prediksi kegagalan mesin industri — Proyek LapisAI × PNJ",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ────────────────────────────────────────────────────────────────
# SCHEMAS
# ────────────────────────────────────────────────────────────────
class SensorInput(BaseModel):
    machine_id: str = Field(..., example="M-01")
    temperature: float = Field(..., example=75.5)
    vibration: float = Field(..., ge=0, example=0.6)
    pressure: float = Field(..., example=102.3)
    rpm: int = Field(..., example=2400)
    power_consumption: float = Field(..., example=78.0)
    noise_level: float = Field(..., example=71.0)
    humidity: float = Field(..., example=55.0)
    operating_hours: float = Field(..., example=1200.0)

    # Rolling (opsional)
    temperature_roll_mean_24h: Optional[float] = None
    temperature_roll_std_24h: Optional[float] = None
    temperature_roll_max_24h: Optional[float] = None
    temperature_roll_min_24h: Optional[float] = None
    vibration_roll_std_24h: Optional[float] = None
    vibration_roll_min_24h: Optional[float] = None
    pressure_roll_std_24h: Optional[float] = None
    pressure_roll_max_24h: Optional[float] = None
    pressure_roll_min_24h: Optional[float] = None
    rpm_roll_mean_24h: Optional[float] = None
    rpm_roll_std_24h: Optional[float] = None
    rpm_roll_max_24h: Optional[float] = None
    rpm_roll_min_24h: Optional[float] = None
    power_consumption_roll_mean_24h: Optional[float] = None
    power_consumption_roll_std_24h: Optional[float] = None
    power_consumption_roll_max_24h: Optional[float] = None
    power_consumption_roll_min_24h: Optional[float] = None
    noise_level_roll_std_24h: Optional[float] = None
    noise_level_roll_max_24h: Optional[float] = None
    noise_level_roll_min_24h: Optional[float] = None
    humidity_roll_mean_24h: Optional[float] = None
    humidity_roll_std_24h: Optional[float] = None
    humidity_roll_max_24h: Optional[float] = None
    humidity_roll_min_24h: Optional[float] = None
    operating_hours_roll_std_24h: Optional[float] = None

    # Delta (opsional)
    temperature_delta: Optional[float] = None
    vibration_delta: Optional[float] = None
    pressure_delta: Optional[float] = None
    rpm_delta: Optional[float] = None
    power_consumption_delta: Optional[float] = None
    noise_level_delta: Optional[float] = None
    humidity_delta: Optional[float] = None
    operating_hours_delta: Optional[float] = None

    # Lag (opsional)
    temperature_lag_1h: Optional[float] = None
    temperature_lag_2h: Optional[float] = None
    temperature_lag_3h: Optional[float] = None
    temperature_lag_6h: Optional[float] = None
    temperature_lag_12h: Optional[float] = None
    temperature_lag_24h: Optional[float] = None
    vibration_lag_1h: Optional[float] = None
    vibration_lag_2h: Optional[float] = None
    vibration_lag_3h: Optional[float] = None
    vibration_lag_6h: Optional[float] = None
    vibration_lag_12h: Optional[float] = None
    vibration_lag_24h: Optional[float] = None
    pressure_lag_1h: Optional[float] = None
    pressure_lag_2h: Optional[float] = None
    pressure_lag_3h: Optional[float] = None
    pressure_lag_6h: Optional[float] = None
    pressure_lag_12h: Optional[float] = None
    pressure_lag_24h: Optional[float] = None
    rpm_lag_1h: Optional[float] = None
    rpm_lag_2h: Optional[float] = None
    rpm_lag_3h: Optional[float] = None
    rpm_lag_6h: Optional[float] = None
    rpm_lag_12h: Optional[float] = None
    rpm_lag_24h: Optional[float] = None

    # Interaksi
    pressure_x_rpm: Optional[float] = None
    temp_x_pressure: Optional[float] = None
    power_per_rpm: Optional[float] = None
    fft_vib_dominant_freq: Optional[float] = None

    # Time
    hour_of_day: Optional[int] = None
    day_of_week: Optional[int] = None
    day_of_month: Optional[int] = None
    month: Optional[int] = None
    is_weekend: Optional[int] = None
    is_night: Optional[int] = None

    # NLP / maintenance
    last_maintenance_type: Optional[float] = None
    last_maintenance_severity_score: Optional[float] = None
    last_problem_category: Optional[float] = None
    last_has_urgent_flag: Optional[float] = None
    last_technical_term_count: Optional[float] = None
    last_problem_keyword_count: Optional[float] = None
    last_action_keyword_count: Optional[float] = None
    days_since_last_maintenance: Optional[float] = None

    threshold: Optional[float] = Field(None, ge=0.0, le=1.0, example=0.20)


# ────────────────────────────────────────────────────────────────
# FEATURE BUILDER (dari main.py lama Anda)
# ────────────────────────────────────────────────────────────────
def build_feature_vector(data: SensorInput) -> np.ndarray:
    """Bangun vektor 83 fitur dari input sensor."""
    now = datetime.now()
    d = data.dict()

    # Time features
    d["hour_of_day"] = d["hour_of_day"] if d["hour_of_day"] is not None else now.hour
    d["day_of_week"] = d["day_of_week"] if d["day_of_week"] is not None else now.weekday()
    d["day_of_month"] = d["day_of_month"] if d["day_of_month"] is not None else now.day
    d["month"] = d["month"] if d["month"] is not None else now.month
    d["is_weekend"] = d["is_weekend"] if d["is_weekend"] is not None else int(now.weekday() >= 5)
    d["is_night"] = d["is_night"] if d["is_night"] is not None else int(now.hour >= 22 or now.hour <= 5)

    # Interaction features
    d["pressure_x_rpm"] = d["pressure_x_rpm"] or (d["pressure"] * d["rpm"])
    d["temp_x_pressure"] = d["temp_x_pressure"] or (d["temperature"] * d["pressure"])
    d["power_per_rpm"] = d["power_per_rpm"] or (d["power_consumption"] / (d["rpm"] + 1))
    d["fft_vib_dominant_freq"] = d["fft_vib_dominant_freq"] or 0.0

    # Rolling defaults
    rolling_defaults = {
        "temperature_roll_mean_24h": d["temperature"],
        "temperature_roll_std_24h": 0.0,
        "temperature_roll_max_24h": d["temperature"],
        "temperature_roll_min_24h": d["temperature"],
        "vibration_roll_std_24h": 0.0,
        "vibration_roll_max_24h": d["vibration"],
        "vibration_roll_min_24h": d["vibration"],
        "pressure_roll_std_24h": 0.0,
        "pressure_roll_max_24h": d["pressure"],
        "pressure_roll_min_24h": d["pressure"],
        "rpm_roll_mean_24h": d["rpm"],
        "rpm_roll_std_24h": 0.0,
        "rpm_roll_max_24h": d["rpm"],
        "rpm_roll_min_24h": d["rpm"],
        "power_consumption_roll_mean_24h": d["power_consumption"],
        "power_consumption_roll_std_24h": 0.0,
        "power_consumption_roll_max_24h": d["power_consumption"],
        "power_consumption_roll_min_24h": d["power_consumption"],
        "noise_level_roll_std_24h": 0.0,
        "noise_level_roll_max_24h": d["noise_level"],
        "noise_level_roll_min_24h": d["noise_level"],
        "humidity_roll_mean_24h": d["humidity"],
        "humidity_roll_std_24h": 0.0,
        "humidity_roll_max_24h": d["humidity"],
        "humidity_roll_min_24h": d["humidity"],
        "operating_hours_roll_std_24h": 0.0,
    }
    for k, v in rolling_defaults.items():
        if d.get(k) is None:
            d[k] = v

    # Delta & lag defaults
    for feat in FEATURES:
        if d.get(feat) is None:
            d[feat] = 0.0

    # NLP defaults
    # NLP defaults — disesuaikan dengan distribusi training (bukan -1)
    # Rationale: -1 adalah outlier value yang trigger false positive di IForest.
    # Pakai nilai realistic dari training distribution.
    nlp_defaults = {
        "last_maintenance_type": 2,                  # 2 = Preventive (paling umum)
        "last_maintenance_severity_score": 1,        # Low severity
        "last_problem_category": 0,                  # Topic 0 (paling umum)
        "last_has_urgent_flag": 0,                   # Tidak urgent
        "last_technical_term_count": 5,              # Median count
        "last_problem_keyword_count": 2,             # Median
        "last_action_keyword_count": 3,              # Median
        "days_since_last_maintenance": 30,           # Average gap antar maintenance
    }
    for k, v in nlp_defaults.items():
        if d.get(k) in [None, 0.0]:
            d[k] = v

    vec = np.array([d[f] for f in FEATURES], dtype=float).reshape(1, -1)
    return vec


def calculate_display_rul(rul_hours: float, fail_prob: float) -> float:
    """
    Hybrid RUL: gabungkan output Task C + Task A.
    Jika Task C stuck di cap 168h, estimasi dari fail_prob Task A.
    """
    # Jika Task C prediksi spesifik (< 167.5) → pakai langsung
    if rul_hours < 167.5:
        return round(rul_hours, 2)

    # Task C stuck di 168 → estimasi dari fail_prob Task A
    if fail_prob >= 0.50:
        # Kritis: estimasi 0–72 jam
        estimated = (1.0 - fail_prob) * 144.0
    elif fail_prob >= 0.20:
        # Warning: estimasi 72–168 jam
        estimated = 72.0 + (1.0 - fail_prob) * 120.0
    else:
        # Aman: lebih dari 7 hari → return 999 sebagai sentinel
        return 999.0

    return round(estimated, 2)


def predict_rul_from_vector(vec_83: np.ndarray) -> float:
    """Predict RUL dari single feature vector (replikasi jadi sequence 24-step)."""
    # Pad/truncate ke n_features LSTM
    if vec_83.shape[1] < LSTM_N_FEATURES:
        pad = np.zeros((1, LSTM_N_FEATURES - vec_83.shape[1]))
        vec_lstm = np.hstack([vec_83, pad])
    else:
        vec_lstm = vec_83[:, :LSTM_N_FEATURES]

    # Replikasi jadi sequence 24-step
    seq = np.tile(vec_lstm, (SEQ_LENGTH, 1))
    seq_scaled = scaler_rul_X.transform(seq).reshape(1, SEQ_LENGTH, -1)

    rul_scaled = float(lstm_rul.predict(seq_scaled, verbose=0)[0, 0])
    rul_hours = float(scaler_rul_y.inverse_transform([[rul_scaled]])[0, 0])
    rul_hours = rul_hours + RUL_BIAS  # bias correction
    return float(np.clip(rul_hours, 0, RUL_CAP))





def predict_anomaly_from_vector(vec_83: np.ndarray) -> tuple:
    """Predict anomaly score dari single vector. Return (score, is_anomaly)."""
    # Pad ke n_features yang diharapkan oleh imputer/scaler
    if vec_83.shape[1] < IFOREST_N_FEATURES:
        pad = np.zeros((1, IFOREST_N_FEATURES - vec_83.shape[1]))
        vec_anom = np.hstack([vec_83, pad])
    else:
        vec_anom = vec_83[:, :IFOREST_N_FEATURES]

    # Pipeline 2-step: imputer → scaler
    vec_imputed = iforest_imputer.transform(vec_anom)
    vec_anom_scaled = scaler_iforest.transform(vec_imputed)

    # IsolationForest: higher score → more anomalous
    score = float(-iforest.score_samples(vec_anom_scaled)[0])
    is_anom = int(score >= IFOREST_THRESHOLD)
    return score, is_anom


def _get_recommendation(will_fail: bool, health_label: str, proba: float,
                         is_anomaly: int = 0, rul_hours: float = 168) -> str:
    if will_fail and health_label == "Critical":
        return "⛔ IMMEDIATELY perform emergency maintenance! The risk of failure is very high.."
    elif will_fail and rul_hours < 48:
        return f"🚨 Maintenance in {rul_hours:.0f} hour. The risk of failure is very close."
    elif will_fail:
        return "⚠️ Schedule maintenance within the next 7 dayst."
    elif is_anomaly:
        return "🔍 Abnormal sensor pattern detected. Further inspection is recommended.."
    elif health_label == "Warning":
        return "🔔 Monitor machines more closely. Consider preventive maintenance.."
    else:
        return "✅ The machine is in normal condition. Continue with the routine maintenance schedule.."


# ────────────────────────────────────────────────────────────────
# ENDPOINTS — INFO
# ────────────────────────────────────────────────────────────────
@app.get("/")
def root():
    return {
        "name": "Predictive Maintenance API",
        "version": "2.0.0",
        "models_loaded": ["TaskA_RF", "TaskB_RF", "TaskC_LSTM", "IsolationForest"],
        "endpoints": {
            "POST /predict": "Task A + B (failure + health)",
            "POST /predict/failure": "Task A only",
            "POST /predict/health": "Task B only",
            "POST /predict/rul": "Task C only (LSTM RUL)",
            "POST /predict/anomaly": "Anomaly detection (IsolationForest)",
            "POST /predict/full": "Semua 4 task sekaligus ⭐",
            "GET /predict/{machine_id}": "Auto-fetch dari Supabase & predict full",
            "GET /dashboard/summary": "KPI summary semua mesin",
            "GET /model/info": "Info model & threshold",
            "GET /health": "Cek API status",
        },
    }


@app.get("/health")
def health_check():
    return {"status": "ok", "timestamp": datetime.now().isoformat()}


@app.get("/model/info")
def model_info():
    return {
        "task_A": {
            "description": "Prediksi kegagalan dalam 7 hari (binary)",
            "model": metadata["task_A"]["model"],
            "roc_auc": metadata["task_A"]["roc_auc"],
            "pr_auc": metadata["task_A"]["pr_auc"],
            "threshold_default": THRESHOLD_DEF,
            "threshold_recall80": THRESHOLD_R80,
        },
        "task_B": {
            "description": "Status kesehatan (Healthy/Warning/Critical)",
            "model": metadata["task_B"]["model"],
            "macro_f1": metadata["task_B"]["macro_f1"],
            "labels": HEALTH_LABELS,
        },
        "task_C": {
            "description": "Remaining Useful Life (jam, cap 168h)",
            "model": "LSTM 128-64-32-16-1",
            "rul_cap_hours": RUL_CAP,
            "bias_correction": RUL_BIAS,
            "seq_length": SEQ_LENGTH,
        },
        "anomaly": {
            "description": "Unsupervised anomaly detection",
            "model": "IsolationForest (n_estimators=200)",
            "threshold": IFOREST_THRESHOLD,
        },
        "n_features": metadata["n_features"],
    }


# ────────────────────────────────────────────────────────────────
# ENDPOINTS — PREDICTION (POST, individual task)
# ────────────────────────────────────────────────────────────────
@app.post("/predict")
def predict_all(data: SensorInput):
    """Task A + Task B."""
    try:
        vec = build_feature_vector(data)
        vec_scaled = scaler.transform(vec)
        threshold = data.threshold if data.threshold is not None else THRESHOLD_DEF

        proba_A = float(model_A.predict_proba(vec_scaled)[0][1])
        will_fail = bool(proba_A >= threshold)
        risk_level = "HIGH" if proba_A >= 0.50 else "MEDIUM" if proba_A >= 0.20 else "LOW"

        pred_B = int(model_B.predict(vec_scaled)[0])
        proba_B = model_B.predict_proba(vec_scaled)[0].tolist()
        health_label = HEALTH_LABELS[str(pred_B)]

        return {
            "machine_id": data.machine_id,
            "timestamp": datetime.now().isoformat(),
            "task_A": {
                "will_fail_within_7days": will_fail,
                "failure_probability": round(proba_A, 4),
                "risk_level": risk_level,
                "threshold_used": threshold,
            },
            "task_B": {
                "health_status": pred_B,
                "health_label": health_label,
                "probabilities": {
                    "Healthy": round(proba_B[0], 4),
                    "Warning": round(proba_B[1], 4),
                    "Critical": round(proba_B[2], 4),
                },
            },
            "recommendation": _get_recommendation(will_fail, health_label, proba_A),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/failure")
def predict_failure(data: SensorInput):
    """Task A only."""
    try:
        vec = build_feature_vector(data)
        vec_scaled = scaler.transform(vec)
        threshold = data.threshold if data.threshold is not None else THRESHOLD_DEF
        proba_A = float(model_A.predict_proba(vec_scaled)[0][1])
        will_fail = bool(proba_A >= threshold)

        return {
            "machine_id": data.machine_id,
            "timestamp": datetime.now().isoformat(),
            "will_fail_within_7days": will_fail,
            "failure_probability": round(proba_A, 4),
            "risk_level": "HIGH" if proba_A >= 0.50 else "MEDIUM" if proba_A >= 0.20 else "LOW",
            "threshold_used": threshold,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/health")
def predict_health(data: SensorInput):
    """Task B only."""
    try:
        vec = build_feature_vector(data)
        vec_scaled = scaler.transform(vec)
        pred_B = int(model_B.predict(vec_scaled)[0])
        proba_B = model_B.predict_proba(vec_scaled)[0].tolist()

        return {
            "machine_id": data.machine_id,
            "timestamp": datetime.now().isoformat(),
            "health_status": pred_B,
            "health_label": HEALTH_LABELS[str(pred_B)],
            "probabilities": {
                "Healthy": round(proba_B[0], 4),
                "Warning": round(proba_B[1], 4),
                "Critical": round(proba_B[2], 4),
            },
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/rul")
def predict_rul(data: SensorInput):
    """Task C only — Remaining Useful Life (LSTM)."""
    try:
        vec = build_feature_vector(data)
        rul_hours = predict_rul_from_vector(vec)

        if rul_hours < 24:
            urgency = "IMMEDIATE"
        elif rul_hours < 72:
            urgency = "CRITICAL"
        elif rul_hours < 168:
            urgency = "WARNING"
        else:
            urgency = "HEALTHY"

        return {
            "machine_id": data.machine_id,
            "timestamp": datetime.now().isoformat(),
            "rul_hours": round(rul_hours, 2),
            "rul_days": round(rul_hours / 24, 2),
            "rul_capped": rul_hours >= RUL_CAP,
            "urgency": urgency,
            "bias_correction_applied": round(RUL_BIAS, 2),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/anomaly")
def predict_anomaly(data: SensorInput):
    """Anomaly detection (IsolationForest)."""
    try:
        vec = build_feature_vector(data)
        score, is_anom = predict_anomaly_from_vector(vec)

        return {
            "machine_id": data.machine_id,
            "timestamp": datetime.now().isoformat(),
            "anomaly_score": round(score, 4),
            "is_anomaly": bool(is_anom),
            "threshold_used": round(IFOREST_THRESHOLD, 4),
            "severity": "HIGH" if score >= IFOREST_THRESHOLD * 1.5
                        else "MEDIUM" if is_anom else "NONE",
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/full")
def predict_full(data: SensorInput):
    """⭐ Predict semua 4 task sekaligus."""
    try:
        vec = build_feature_vector(data)
        vec_scaled = scaler.transform(vec)
        threshold = data.threshold if data.threshold is not None else THRESHOLD_DEF

        # Task A
        proba_A = float(model_A.predict_proba(vec_scaled)[0][1])
        will_fail = bool(proba_A >= threshold)

        # Task B
        pred_B = int(model_B.predict(vec_scaled)[0])
        proba_B = model_B.predict_proba(vec_scaled)[0].tolist()
        health_label = HEALTH_LABELS[str(pred_B)]

        # Task C
        rul_hours_raw = predict_rul_from_vector(vec)      # 0–168, untuk health score
        rul_hours = calculate_display_rul(rul_hours_raw, proba_A)  # untuk display

        # Anomaly
        anom_score, is_anom = predict_anomaly_from_vector(vec)

        # Overall health score (0-100)
        health_score = int(np.clip(
            100 * (1 - proba_A) * (rul_hours_raw / RUL_CAP) * (1 - 0.2 * is_anom),
            0, 100
        ))

        return {
            "machine_id": data.machine_id,
            "timestamp": datetime.now().isoformat(),
            "overall_health_score": health_score,
            "task_A": {
                "will_fail_within_7days": will_fail,
                "failure_probability": round(proba_A, 4),
                "risk_level": "HIGH" if proba_A >= 0.50 else "MEDIUM" if proba_A >= 0.20 else "LOW",
            },
            "task_B": {
                "health_status": pred_B,
                "health_label": health_label,
                "probabilities": {
                    "Healthy": round(proba_B[0], 4),
                    "Warning": round(proba_B[1], 4),
                    "Critical": round(proba_B[2], 4),
                },
            },
            "task_C": {
                "rul_hours": round(rul_hours, 2) if rul_hours < 999 else 999,
                "rul_days": round(rul_hours / 24, 2) if rul_hours < 999 else 999,
                "rul_capped": rul_hours >= 999,
            },
            "anomaly": {
                "score": round(anom_score, 4),
                "is_anomaly": bool(is_anom),
            },
            "recommendation": _get_recommendation(will_fail, health_label, proba_A, is_anom, rul_hours),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ────────────────────────────────────────────────────────────────
# ENDPOINTS — SUPABASE-INTEGRATED (GET, auto-fetch)
# ────────────────────────────────────────────────────────────────
def fetch_latest_sensor(machine_id: str) -> dict:
    """Ambil reading sensor terbaru dari Supabase untuk satu mesin."""
    if supabase is None:
        raise HTTPException(503, "Supabase not configured. Set SUPABASE_URL & SUPABASE_SERVICE_KEY in .env")

    resp = (
        supabase.table("sensor_readings")
        .select("*")
        .eq("machine_id", machine_id)
        .order("timestamp", desc=True)
        .limit(1)
        .execute()
    )
    if not resp.data:
        raise HTTPException(404, f"No data for {machine_id}")
    return resp.data[0]


@app.get("/predict/{machine_id}")
def predict_by_machine_id(machine_id: str):
    """Auto-fetch reading terbaru dari Supabase → predict full."""
    latest = fetch_latest_sensor(machine_id)

    sensor_input = SensorInput(
        machine_id=machine_id,
        temperature=latest.get("temperature", 0),
        vibration=latest.get("vibration", 0),
        pressure=latest.get("pressure", 0),
        rpm=int(latest.get("rpm", 0)),
        power_consumption=latest.get("power_consumption", 0),
        noise_level=latest.get("noise_level", 0),
        humidity=latest.get("humidity", 0),
        operating_hours=latest.get("operating_hours", 0),
    )
    return predict_full(sensor_input)


@app.get("/dashboard/summary")
def dashboard_summary():
    """KPI summary semua mesin (untuk kartu di dashboard)."""
    if supabase is None:
        raise HTTPException(503, "Supabase not configured")

    machines = [f"M-{i:02d}" for i in range(1, 21)]
    results = []
    errors = []

    for mid in machines:
        try:
            result = predict_by_machine_id(mid)
            results.append(result)
        except HTTPException as e:
            errors.append({"machine_id": mid, "error": e.detail})
        except Exception as e:
            errors.append({"machine_id": mid, "error": str(e)})

    if not results:
        return {"error": "No predictions available", "errors": errors}

    most_critical = min(results, key=lambda x: x["overall_health_score"])

    return {
        "total_machines": len(results),
        "critical_count": sum(1 for r in results if r["task_B"]["health_status"] == 2),
        "warning_count": sum(1 for r in results if r["task_B"]["health_status"] == 1),
        "will_fail_count": sum(1 for r in results if r["task_A"]["will_fail_within_7days"]),
        "anomaly_count": sum(1 for r in results if r["anomaly"]["is_anomaly"]),
        "avg_health_score": int(np.mean([r["overall_health_score"] for r in results])),
        "most_critical_machine": most_critical["machine_id"],
        "most_critical_score": most_critical["overall_health_score"],
        "updated_at": datetime.now().isoformat(),
        "machines": results,
        "errors": errors if errors else None,
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)