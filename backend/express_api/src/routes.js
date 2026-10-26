import { Router } from "express";
import { NlpModel, RlModel, PredictionLog, NewsHeadline, A2CStrategy } from "./models.js";

export const router = Router();

router.get("/health", (req, res) => res.json({ status: "ok" }));

router.get("/nlp-models", async (req, res) => {
  const models = await NlpModel.find().sort({ f1_weighted: -1 }).lean();
  res.json({ models });
});

router.get("/rl-models", async (req, res) => {
  const models = await RlModel.find().lean();
  res.json({ models });
});

// Called by scripts/collect_rl_results.py once a training notebook finishes.
router.put("/rl-models/:agent", async (req, res) => {
  const doc = await RlModel.findOneAndUpdate(
    { agent: req.params.agent },
    { ...req.body, updatedAt: new Date() },
    { new: true, upsert: true }
  );
  res.json(doc);
});

// Called by the dashboard's live sentiment-demo page after it gets a prediction from the
// FastAPI ML service — this just logs it so a "recent predictions" feed has something real
// to show, rather than the dashboard talking to Mongo for the inference itself (that stays
// in FastAPI/PyTorch, where the model actually lives).
router.post("/predictions", async (req, res) => {
  const { text, label, confidence, probabilities } = req.body;
  if (!text || !label || confidence === undefined) {
    return res.status(400).json({ error: "text, label, and confidence are required" });
  }
  const doc = await PredictionLog.create({ text, label, confidence, probabilities });
  res.status(201).json(doc);
});

router.get("/predictions", async (req, res) => {
  const limit = Math.min(parseInt(req.query.limit) || 20, 100);
  const docs = await PredictionLog.find().sort({ createdAt: -1 }).limit(limit).lean();
  res.json({ predictions: docs });
});

router.get("/news", async (req, res) => {
  const filter = req.query.ticker ? { ticker: req.query.ticker } : {};
  const headlines = await NewsHeadline.find(filter).sort({ date: -1 }).lean();
  res.json({ headlines });
});

router.get("/strategy/a2c", async (req, res) => {
  const strategy = await A2CStrategy.findOne().lean();
  if (!strategy) return res.status(404).json({ error: "A2C strategy not seeded yet" });
  res.json(strategy);
});
