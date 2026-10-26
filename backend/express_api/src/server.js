import express from "express";
import cors from "cors";
import "dotenv/config";

import { connectDB } from "./db.js";
import { seedIfEmpty } from "./seed.js";
import { router } from "./routes.js";

const PORT = process.env.PORT || 4000;

async function main() {
  await connectDB();
  await seedIfEmpty();

  const app = express();
  app.use(cors());
  app.use(express.json());
  app.use("/api", router);

  app.listen(PORT, () => console.log(`Express API listening on http://localhost:${PORT}`));
}

main().catch((err) => {
  console.error("Failed to start server:", err);
  process.exit(1);
});
