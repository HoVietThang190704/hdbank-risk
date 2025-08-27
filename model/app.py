from fastapi import FastAPI
from pydantic import BaseModel, Field
import time, random

app = FastAPI(title="Risk Model Service", version="0.1.0")

class Features(BaseModel):
    customer_id: str
    age: int = Field(ge=18, le=100)
    income: float = Field(ge=0)
    liabilities: float = Field(ge=0)
    credit_history_months: int = Field(ge=0)

@app.get("/v1/healthz")
def healthz():
    return {"ok": True, "service": "model"}

@app.post("/v1/predict")
def predict(f: Features):
    t0 = time.time()
    base = 0.03
    income_factor = 1.0 if f.income <= 0 else min(1.0, 10_000_000 / (f.income + 1))
    debt_ratio = 0 if f.income == 0 else f.liabilities / (f.income + 1)
    history_bonus = max(0, (12 - min(120, f.credit_history_months))) / 120.0
    noise = (random.random() - 0.5) * 0.01
    pd = base + 0.02*income_factor + 0.03*debt_ratio + 0.02*history_bonus + noise
    pd = max(0.0001, min(0.9999, pd))
    top_reasons = [
        {"feature": "liabilities/income", "impact": round(0.03 * debt_ratio, 6)},
        {"feature": "credit_history_months", "impact": round(0.02 * history_bonus, 6)},
        {"feature": "income", "impact": round(0.02 * income_factor, 6)},
    ]
    return {
        "pd": float(pd),
        "top_reasons": top_reasons,
        "model_version": "pd_mock_0.1.0",
        "latency_ms": int((time.time() - t0) * 1000)
    }
