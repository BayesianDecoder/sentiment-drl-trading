/**
 * Seed data for NlpModel (Part A's real final benchmark results) and RlModel placeholders
 * (Part B, filled in by update_rl_results.js once training finishes).
 *
 * Called from server.js's own startup (NOT run as a separate script) — mongodb-memory-server
 * spins up a fresh, separate in-memory instance per process, so a standalone seed script would
 * seed data that vanishes the moment the real server starts against a different instance.
 */
import fs from "fs";
import path from "path";
import { fileURLToPath } from "url";
import { NlpModel, RlModel, NewsHeadline, A2CStrategy } from "./models.js";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const PROCESSED_DIR = path.join(__dirname, "..", "..", "..", "data", "processed");

const NLP_RESULTS = [
  { name: "Logistic Regression (tuned)", accuracy: 0.81, f1_weighted: 0.81, category: "classical" },
  { name: "SVM (tuned)", accuracy: 0.81, f1_weighted: 0.81, category: "classical" },
  { name: "Random Forest (tuned)", accuracy: 0.78, f1_weighted: 0.78, category: "classical" },
  { name: "LightGBM (tuned)", accuracy: 0.79, f1_weighted: 0.79, category: "classical" },
  { name: "XGBoost (tuned)", accuracy: 0.79, f1_weighted: 0.79, category: "classical" },
  { name: "TextCNN (tuned)", accuracy: 0.7639, f1_weighted: 0.7636, category: "deep_learning" },
  { name: "BiLSTM (tuned)", accuracy: 0.7731, f1_weighted: 0.7733, category: "deep_learning" },
  { name: "CNN-BiLSTM-Attention (tuned)", accuracy: 0.7646, f1_weighted: 0.7642, category: "deep_learning" },
  { name: "Transformer (scratch, tuned)", accuracy: 0.7773, f1_weighted: 0.7771, category: "deep_learning" },
  { name: "FinBERT + LoRA", accuracy: 0.8654, f1_weighted: 0.8652, category: "transformer" },
];

const RL_AGENTS = ["PPO", "A2C", "DDPG", "SAC", "TD3"];

export async function seedIfEmpty() {
  if ((await NlpModel.countDocuments()) === 0) {
    await NlpModel.insertMany(NLP_RESULTS);
    console.log(`Seeded ${NLP_RESULTS.length} NLP model results`);
  }
  if ((await RlModel.countDocuments()) === 0) {
    await RlModel.insertMany(RL_AGENTS.map((agent) => ({ agent, status: "pending" })));
    console.log(`Seeded ${RL_AGENTS.length} RL agent placeholders (status=pending)`);
  }

  if ((await NewsHeadline.countDocuments()) === 0) {
    const newsPath = path.join(PROCESSED_DIR, "news_headlines.json");
    if (fs.existsSync(newsPath)) {
      const headlines = JSON.parse(fs.readFileSync(newsPath, "utf-8"));
      await NewsHeadline.insertMany(headlines);
      console.log(`Seeded ${headlines.length} news headlines`);
    } else {
      console.log("news_headlines.json not found -- run scripts/fetch_news_headlines.py first");
    }
  }

  if ((await A2CStrategy.countDocuments()) === 0) {
    const strategyPath = path.join(PROCESSED_DIR, "a2c_strategy.json");
    if (fs.existsSync(strategyPath)) {
      const strategy = JSON.parse(fs.readFileSync(strategyPath, "utf-8"));
      await A2CStrategy.create(strategy);
      console.log(`Seeded A2C strategy timeline (${strategy.days.length} days)`);
    } else {
      console.log("a2c_strategy.json not found -- run scripts/generate_a2c_strategy.py first");
    }
  }
}
