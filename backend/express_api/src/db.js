import mongoose from "mongoose";
import { MongoMemoryServer } from "mongodb-memory-server";

// No system MongoDB was available in this environment (no Homebrew/package manager to install
// it), so this uses mongodb-memory-server, which downloads and runs a REAL mongod binary
// locally — same MongoDB, just without a separate install step. Data lives in a temp dir and
// resets when the process restarts; swap this for a real MONGODB_URI env var to persist.
let memoryServer = null;

export async function connectDB() {
  if (process.env.MONGODB_URI) {
    await mongoose.connect(process.env.MONGODB_URI);
    console.log("Connected to MongoDB at", process.env.MONGODB_URI);
    return;
  }

  memoryServer = await MongoMemoryServer.create();
  const uri = memoryServer.getUri();
  await mongoose.connect(uri);
  console.log("Connected to in-memory MongoDB at", uri);
}

export async function disconnectDB() {
  await mongoose.disconnect();
  if (memoryServer) await memoryServer.stop();
}
