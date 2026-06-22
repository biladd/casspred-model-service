# Predictive Maintenance API

FastAPI backend untuk model ML prediksi kegagalan mesin.

## Struktur Folder

```
fastapi_app/
├── main.py              # API utama
├── requirements.txt     # Dependencies
├── Dockerfile           # Untuk deployment
└── models/
    ├── final_model_taskA.pkl
    ├── final_model_taskB.pkl
    ├── scaler.pkl
    └── final_metadata.json
```

## Cara Jalankan Lokal

```bash
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

Buka browser: http://localhost:8000/docs

## Endpoints

| Method | URL | Keterangan |
|--------|-----|------------|
| GET | `/` | Info API |
| GET | `/health` | Cek status API |
| GET | `/model/info` | Info model & threshold |
| POST | `/predict` | Prediksi lengkap (Task A + B) |
| POST | `/predict/failure` | Hanya prediksi kegagalan 7 hari |
| POST | `/predict/health` | Hanya status kesehatan mesin |

## Contoh Request (dari Next.js)

```javascript
const response = await fetch("https://your-api.railway.app/predict", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({
    machine_id: "M-01",
    temperature: 85.5,
    vibration: 1.1,
    pressure: 110.0,
    rpm: 2400,
    power_consumption: 82.0,
    noise_level: 75.0,
    humidity: 60.0,
    operating_hours: 2000.0
  })
});

const result = await response.json();
console.log(result);
```

## Contoh Response

```json
{
  "machine_id": "M-01",
  "timestamp": "2025-05-12T10:00:00",
  "task_A": {
    "will_fail_within_7days": true,
    "failure_probability": 0.67,
    "risk_level": "HIGH",
    "threshold_used": 0.20
  },
  "task_B": {
    "health_status": 2,
    "health_label": "Critical",
    "probabilities": {
      "Healthy": 0.02,
      "Warning": 0.10,
      "Critical": 0.88
    }
  },
  "recommendation": "⛔ SEGERA lakukan maintenance darurat! Risiko kegagalan sangat tinggi."
}
```

## Deploy ke Railway

1. Push folder ini ke GitHub
2. Buka railway.app → New Project → Deploy from GitHub
3. Pilih repo ini
4. Railway otomatis detect Dockerfile dan deploy
5. Copy URL yang diberikan Railway → pakai di Next.js

## Integrasi Supabase

Setelah dapat hasil prediksi dari API, simpan ke Supabase:

```javascript
// Di Next.js
const prediction = await fetch("/api/predict", ...).then(r => r.json());

await supabase.from("predictions").insert({
  machine_id: prediction.machine_id,
  will_fail: prediction.task_A.will_fail_within_7days,
  health_label: prediction.task_B.health_label,
  failure_probability: prediction.task_A.failure_probability,
  created_at: new Date().toISOString()
});
```
