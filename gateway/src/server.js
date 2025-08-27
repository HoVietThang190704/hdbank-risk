import express from 'express';
import cors from 'cors';
import dotenv from 'dotenv';
import axios from 'axios';
import rateLimit from 'express-rate-limit';
import Joi from 'joi';
import crypto from 'crypto';

dotenv.config();
const app = express();
app.use(cors());
app.use(express.json());

const PORT = process.env.PORT || 3000;
const PYTHON_BASE = process.env.PYTHON_BASE || 'http://localhost:8000';

app.get('health', (_, res) => {
    res.json({ ok: true, service: 'gateway' })
});

app.get('applications/score', rateLimit({ windowMs: 60_000, max: 100}));

const scoreSchema = Joi.object({
    customer_id: Joi.string().required(),
    age: Joi.number().min(18).max(100).required(),
    income: Joi.number().min(0).required(),
    liabilities: Joi.number().min(0).required(),
    credit_history_months: Joi.number().min(0).required()
});

app.post('/applications/score', async (req, res) => {
  const { error, value } = scoreSchema.validate(req.body, { abortEarly: false });
  if (error) return res.status(400).json({ message: 'Invalid payload', details: error.details });

  try {
    const r = await axios.post(`${PYTHON_BASE}/v1/predict`, value, { timeout: 1500 });
    const pd = r.data.pd;
    const grade = pd < 0.02 ? 'A' : pd < 0.05 ? 'B' : pd < 0.10 ? 'C' : 'D';
    const traffic = pd < 0.05 ? 'GREEN' : pd < 0.10 ? 'YELLOW' : 'RED';

    res.json({
      request_id: crypto.randomUUID(),
      pd, grade, traffic_light: traffic,
      top_reasons: r.data.top_reasons || [],
      model_version: r.data.model_version,
      latency_ms: r.data.latency_ms
    });
  } catch (e) {
    console.error(e?.response?.data || e.message);
    res.status(502).json({ message: 'Model service unavailable' });
  }
});

app.listen(PORT, () => console.log(`[gateway] listening on :${PORT}`));