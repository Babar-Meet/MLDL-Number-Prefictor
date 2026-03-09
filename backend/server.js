import "dotenv/config";
import cors from "cors";
import express from "express";
import http from "http";
import axios from "axios";
import { Server } from "socket.io";
import { existsSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import {
  connectToDatabase,
  disconnectFromDatabase,
  getDatabaseState,
  loadHistoryFromDatabase,
  persistHistoryEntry,
} from "./db.js";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const frontendDist = path.resolve(__dirname, "../frontend/dist");
const PORT = Number(process.env.PORT || 4000);
const PYTHON_SERVICE_URL =
  process.env.PYTHON_SERVICE_URL || "http://127.0.0.1:8000";
const MAX_HISTORY = Number(process.env.MAX_HISTORY || 18);
const predictionHistory = [];

const app = express();
const server = http.createServer(app);
const io = new Server(server, {
  cors: {
    origin: "*",
  },
});

app.use(cors());
app.use(express.json({ limit: "5mb" }));

const pythonApi = axios.create({
  baseURL: PYTHON_SERVICE_URL,
  timeout: 15000,
});

async function appendHistory(analysis) {
  const entry = {
    id: `${Date.now()}-${Math.random().toString(36).slice(2, 8)}`,
    createdAt: new Date().toISOString(),
    digit: analysis.prediction.digit,
    confidence: analysis.prediction.confidence,
    probabilities: analysis.prediction.probabilities,
    controls: analysis.controls,
  };

  predictionHistory.unshift(entry);
  if (predictionHistory.length > MAX_HISTORY) {
    predictionHistory.length = MAX_HISTORY;
  }

  try {
    await persistHistoryEntry(entry);
  } catch (error) {
    console.error("Failed to persist prediction history:", error.message);
  }

  return entry;
}

function normaliseError(error) {
  if (error.response?.data) {
    return error.response.data;
  }

  return {
    message: error.message,
  };
}

app.get("/api/health", async (_request, response) => {
  try {
    const [{ data: python }, uptime] = await Promise.all([
      pythonApi.get("/health"),
      Promise.resolve(process.uptime()),
    ]);

    response.json({
      status: "ok",
      python,
      database: getDatabaseState(),
      uptime,
      historySize: predictionHistory.length,
    });
  } catch (error) {
    response.status(502).json({
      status: "degraded",
      error: normaliseError(error),
    });
  }
});

app.get("/api/model", async (_request, response) => {
  try {
    const { data } = await pythonApi.get("/model");
    response.json(data);
  } catch (error) {
    response.status(502).json({ error: normaliseError(error) });
  }
});

app.get("/api/history", (_request, response) => {
  response.json({ history: predictionHistory });
});

app.post("/api/analyze", async (request, response) => {
  try {
    const { data } = await pythonApi.post("/analyze", request.body);
    await appendHistory(data);
    io.emit("prediction-history", predictionHistory);
    response.json({ analysis: data, history: predictionHistory });
  } catch (error) {
    response.status(502).json({ error: normaliseError(error) });
  }
});

app.post("/api/train-step", async (request, response) => {
  try {
    const { data } = await pythonApi.post("/train-step", request.body);
    await appendHistory(data);
    io.emit("prediction-history", predictionHistory);
    io.emit("model-refresh-required");
    response.json({ analysis: data, history: predictionHistory });
  } catch (error) {
    response.status(502).json({ error: normaliseError(error) });
  }
});

io.on("connection", (socket) => {
  socket.emit("prediction-history", predictionHistory);

  socket.on("request-model", async (acknowledge) => {
    try {
      const { data } = await pythonApi.get("/model");
      if (typeof acknowledge === "function") {
        acknowledge({ ok: true, model: data });
      }
    } catch (error) {
      if (typeof acknowledge === "function") {
        acknowledge({ ok: false, error: normaliseError(error) });
      }
    }
  });

  socket.on("analyze-digit", async (payload, acknowledge) => {
    try {
      const { data } = await pythonApi.post("/analyze", payload);
      await appendHistory(data);
      io.emit("prediction-history", predictionHistory);
      socket.emit("analysis-complete", data);
      if (typeof acknowledge === "function") {
        acknowledge({ ok: true, analysis: data, history: predictionHistory });
      }
    } catch (error) {
      if (typeof acknowledge === "function") {
        acknowledge({ ok: false, error: normaliseError(error) });
      }
    }
  });

  socket.on("train-digit", async (payload, acknowledge) => {
    try {
      const { data } = await pythonApi.post("/train-step", payload);
      await appendHistory(data);
      io.emit("prediction-history", predictionHistory);
      io.emit("model-refresh-required");
      socket.emit("analysis-complete", data);
      if (typeof acknowledge === "function") {
        acknowledge({ ok: true, analysis: data, history: predictionHistory });
      }
    } catch (error) {
      if (typeof acknowledge === "function") {
        acknowledge({ ok: false, error: normaliseError(error) });
      }
    }
  });
});

async function bootstrap() {
  try {
    await connectToDatabase();
    const storedHistory = await loadHistoryFromDatabase();
    predictionHistory.splice(0, predictionHistory.length, ...storedHistory);
  } catch (error) {
    console.warn(
      `MongoDB connection unavailable, using in-memory history: ${error.message}`,
    );
  }

  server.listen(PORT, () => {
    console.log(`Express bridge listening on http://localhost:${PORT}`);
    console.log(`Proxying Python ML service at ${PYTHON_SERVICE_URL}`);
    console.log(`Prediction history mode: ${getDatabaseState().mode}`);
  });
}

if (existsSync(frontendDist)) {
  app.use(express.static(frontendDist));

  app.get(/^(?!\/api).*/, (_request, response) => {
    response.sendFile(path.join(frontendDist, "index.html"));
  });
}

process.on("SIGINT", async () => {
  await disconnectFromDatabase();
  process.exit(0);
});

process.on("SIGTERM", async () => {
  await disconnectFromDatabase();
  process.exit(0);
});

bootstrap();
