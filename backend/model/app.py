from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional
import os, json, time
import numpy as np
import joblib

APP_TITLE = "Risk Model Service"
APP_VERSION = "1.0.0"
ARTIFACTS_DIR = os.getenv("ARTIFACTS_DIR", "/models")

REQUIRED_FILES = {
    "model":        "pd_model.pkl",
    "calibrator":   "calibrator.pkl",          # optional nhưng khuyến nghị
    "feature_order":"feature_order.json"
}

app = FastAPI(title=APP_TITLE, version=APP_VERSION)

# ---------------------- Schemas ---------------------- #
class Features(BaseModel):
    customer_id: str
    age: float = Field(ge=18, le=120)
    income: float = Field(ge=0)
    liabilities: float = Field(ge=0)
    credit_history_months: float = Field(ge=0)

class PredictResponse(BaseModel):
    pd: float
    top_reasons: List[dict]
    model_version: str
    latency_ms: int

# ---------------------- Load artifacts ---------------------- #
MODEL = None
CALIB = None
FEATURE_ORDER: Optional[List[str]] = None
METRICS = None

def load_artifacts():
    global MODEL, CALIB, FEATURE_ORDER, METRICS

    # bắt buộc có model + feature_order
    model_path  = os.path.join(ARTIFACTS_DIR, REQUIRED_FILES["model"])
    calib_path  = os.path.join(ARTIFACTS_DIR, REQUIRED_FILES["calibrator"])
    order_path  = os.path.join(ARTIFACTS_DIR, REQUIRED_FILES["feature_order"])
    metrics_path= os.path.join(ARTIFACTS_DIR, "metrics.json")

    missing = []
    if not os.path.isfile(model_path): missing.append(model_path)
    if not os.path.isfile(order_path): missing.append(order_path)
    # calibrator có thể vắng, chỉ cảnh báo
    if missing:
        return {"ok": False, "missing": missing}

    MODEL = joblib.load(model_path)
    FEATURE_ORDER = json.loads(open(order_path, "r").read())
    CALIB = joblib.load(calib_path) if os.path.isfile(calib_path) else None
    METRICS = None
    if os.path.isfile(metrics_path):
        try:
            METRICS = json.loads(open(metrics_path, "r", encoding="utf-8").read())
        except Exception:
            METRICS = None

    return {"ok": True, "missing": []}

LOAD_STATUS = load_artifacts()

# ---------------------- Utilities ---------------------- #
def build_feature_vector(f: Features) -> np.ndarray:
    """
    Sắp xếp features theo feature_order.json
    """
    values = {
        "age": float(f.age),
        "income": float(f.income),
        "liabilities": float(f.liabilities),
        "credit_history_months": float(f.credit_history_months),
    }
    if not FEATURE_ORDER:
        raise RuntimeError("FEATURE_ORDER is None")
    vec = [values[name] for name in FEATURE_ORDER]
    return np.asarray([vec], dtype=float)  # shape (1, n)

def predict_proba(vec: np.ndarray) -> float:
    """
    Ưu tiên dùng calibrator nếu có; nếu không thì dùng model trực tiếp.
    """
    if CALIB is not None:
        proba = CALIB.predict_proba(vec)[:, 1]
    else:
        # MODEL có thể là Pipeline scikit-learn
        proba = MODEL.predict_proba(vec)[:, 1]
    return float(np.clip(proba[0], 1e-6, 1 - 1e-6))

# ---------------------- Endpoints ---------------------- #
@app.get("/v1/healthz")
def healthz():
    return {
        "ok": bool(LOAD_STATUS.get("ok")),
        "service": "model",
        "artifacts_dir": ARTIFACTS_DIR,
        "required": {
            "model": os.path.join(ARTIFACTS_DIR, REQUIRED_FILES["model"]),
            "calibrator": os.path.join(ARTIFACTS_DIR, REQUIRED_FILES["calibrator"]),
            "feature_order": os.path.join(ARTIFACTS_DIR, REQUIRED_FILES["feature_order"]),
        },
        "missing": LOAD_STATUS.get("missing", []),
        "metrics": METRICS or {}
    }

@app.get("/v1/metadata")
def metadata():
    return {
        "model_version": APP_VERSION,
        "feature_order": FEATURE_ORDER,
        "metrics": METRICS
    }

@app.post("/v1/predict", response_model=PredictResponse)
def predict(f: Features):
    if not LOAD_STATUS.get("ok"):
        raise HTTPException(status_code=503, detail={
            "error": "Artifacts not loaded",
            "missing": LOAD_STATUS.get("missing", [])
        })
    t0 = time.time()
    x = build_feature_vector(f)
    pd_hat = predict_proba(x)

    # Gợi ý reason đơn giản dựa trên độ lớn tuyệt đối (placeholder)
    contrib = [
        ("liabilities", float(f.liabilities)),
        ("income", -float(f.income)),
        ("credit_history_months", -float(f.credit_history_months)),
        ("age", -float(f.age)),
    ]
    contrib = sorted(contrib, key=lambda z: abs(z[1]), reverse=True)[:3]
    top_reasons = [{"feature": k, "impact": round(v, 6)} for k, v in contrib]

    return PredictResponse(
        pd=pd_hat,
        top_reasons=top_reasons,
        model_version=f"pd_production_{APP_VERSION}",
        latency_ms=int((time.time() - t0) * 1000),
    )
