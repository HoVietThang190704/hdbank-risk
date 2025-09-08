// gateway/src/server.js
import express from "express";
import axios from "axios";

const app = express();
app.use(express.json());

const TW = process.env.MODEL_TAIWAN_URL || "http://model_taiwan:8000";
const LC = process.env.MODEL_LC_URL || "http://model_lendingclub:8000";
const REFRESH_SEC = Number(process.env.ENSEMBLE_REFRESH_SEC || 600);
let STRATEGY = (process.env.ENSEMBLE_STRATEGY || "weighted").toLowerCase(); // avg|weighted|tw|lc

// ------------------ State ------------------
let weights = { tw: 0.5, lc: 0.5, updated_at: 0 };
let meta = {
  tw: { version: null, flavor: "taiwan", metrics: {} },
  lc: { version: null, flavor: "lendingclub", metrics: {} },
  updated_at: 0
};

// ------------------ Helpers ------------------
function computeWeightsFromAUC(aucTw, aucLc) {
  if (typeof aucTw === "number" && typeof aucLc === "number") {
    const s = Math.max(aucTw, 1e-6) + Math.max(aucLc, 1e-6);
    return {
      tw: Math.max(aucTw, 1e-6) / s,
      lc: Math.max(aucLc, 1e-6) / s
    };
  }
  return { tw: 0.5, lc: 0.5 };
}

async function refreshMetaAndWeights() {
  const [mTw, mLc] = await Promise.all([
    axios.get(`${TW}/v1/metadata`).then(r => r.data).catch(() => null),
    axios.get(`${LC}/v1/metadata`).then(r => r.data).catch(() => null),
  ]);

  if (mTw) {
    meta.tw = {
      version: mTw.model_version || null,
      flavor: mTw.metrics?.flavor || "taiwan",
      metrics: {
        auc_raw: mTw.metrics?.auc_raw ?? null,
        auc_calibrated: mTw.metrics?.auc_calibrated ?? null,
        ks: mTw.metrics?.ks ?? null,
        gini: mTw.metrics?.gini ?? null
      }
    };
  }
  if (mLc) {
    meta.lc = {
      version: mLc.model_version || null,
      flavor: mLc.metrics?.flavor || "lendingclub",
      metrics: {
        auc_raw: mLc.metrics?.auc_raw ?? null,
        auc_calibrated: mLc.metrics?.auc_calibrated ?? null,
        ks: mLc.metrics?.ks ?? null,
        gini: mLc.metrics?.gini ?? null
      }
    };
  }
  meta.updated_at = Date.now();

  const aucTw = meta.tw.metrics.auc_calibrated ?? meta.tw.metrics.auc_raw;
  const aucLc = meta.lc.metrics.auc_calibrated ?? meta.lc.metrics.auc_raw;
  const w = computeWeightsFromAUC(aucTw, aucLc);
  weights = { ...w, updated_at: Date.now() };

  return { meta, weights };
}

// gọi 1 lần khi start và định kỳ
refreshMetaAndWeights().catch(() => {});
setInterval(() => refreshMetaAndWeights().catch(() => {}), REFRESH_SEC * 1000);

const clamp = (p) => Math.min(1 - 1e-9, Math.max(1e-9, p));
const logit = (p) => Math.log(p / (1 - p));
const invlogit = (z) => 1 / (1 + Math.exp(-z));

// ------------------ Endpoints ------------------
app.get("/ensemble/weights", (req, res) => {
  res.json({ strategy: STRATEGY, weights, meta, refresh_sec: REFRESH_SEC });
});

// POST /applications/score?strategy=avg|weighted|tw|lc
app.post("/applications/score", async (req, res) => {
  const payload = req.body;
  const strategy = (req.query.strategy || STRATEGY).toLowerCase();

  try {
    const t0 = Date.now();
    // gọi 2 model song song
    const [tw, lc] = await Promise.all([
      axios.post(`${TW}/v1/predict`, payload).then(r => r.data),
      axios.post(`${LC}/v1/predict`, payload).then(r => r.data),
    ]);

    let pd;
    if (strategy === "tw") pd = tw.pd;
    else if (strategy === "lc") pd = lc.pd;
    else if (strategy === "avg") pd = (tw.pd + lc.pd) / 2;
    else { // 'weighted' (mặc định) - ENSEMBLE THEO LOGIT
      const z =
        weights.tw * logit(clamp(tw.pd)) +
        weights.lc * logit(clamp(lc.pd));
      pd = invlogit(z);
    }

    res.json({
      strategy,
      weights,
      pd,
      components: {
        taiwan: {
          pd: tw.pd,
          version: meta.tw.version,
          flavor: meta.tw.flavor,
          metrics: meta.tw.metrics
        },
        lendingclub: {
          pd: lc.pd,
          version: meta.lc.version,
          flavor: meta.lc.flavor,
          metrics: meta.lc.metrics
        }
      },
      reasons: [...(tw.top_reasons || []), ...(lc.top_reasons || [])].slice(0, 3),
      latency_ms: Date.now() - t0,
      meta_updated_at: meta.updated_at
    });
  } catch (err) {
    console.error(err?.response?.data || err.message);
    res.status(502).json({ error: "Gateway ensemble error", detail: err?.response?.data || err.message });
  }
});

app.listen(3000, () => console.log("Gateway running on :3000"));
