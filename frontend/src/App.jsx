import {
  lazy,
  Suspense,
  useCallback,
  useDeferredValue,
  useEffect,
  useRef,
  useState,
  useTransition,
} from "react";
import { motion, AnimatePresence } from "framer-motion";
import axios from "axios";
import { io } from "socket.io-client";
import {
  Brain,
  Activity,
  Layers,
  Zap,
  Sparkles,
  RotateCcw,
  Eye,
  Box,
  Target,
  Settings2,
  FlaskConical,
  ChevronRight,
  Play,
  Square,
} from "lucide-react";
import "./App.css";
import ActivationCharts from "./components/ActivationCharts";
import DigitCanvas from "./components/DigitCanvas";
import NetworkGraph from "./components/NetworkGraph";
import PredictionHistory from "./components/PredictionHistory";

const NetworkScene3D = lazy(() => import("./components/NetworkScene3D"));

const api = axios.create({
  baseURL: import.meta.env.VITE_API_BASE_URL || "",
});

const segmentMap = {
  0: ["a", "b", "c", "d", "e", "f"],
  1: ["b", "c"],
  2: ["a", "b", "g", "e", "d"],
  3: ["a", "b", "g", "c", "d"],
  4: ["f", "g", "b", "c"],
  5: ["a", "f", "g", "c", "d"],
  6: ["a", "f", "g", "c", "d", "e"],
  7: ["a", "b", "c"],
  8: ["a", "b", "c", "d", "e", "f", "g"],
  9: ["a", "b", "c", "d", "f", "g"],
};

function createEmptyGrid() {
  return Array.from({ length: 28 }, () => Array(28).fill(0));
}

function clamp(value, min, max) {
  return Math.max(min, Math.min(max, value));
}

function paintRect(grid, top, left, height, width, value = 1) {
  const nextGrid = grid.map((row) => [...row]);
  for (let row = top; row < top + height; row += 1) {
    for (let col = left; col < left + width; col += 1) {
      if (row >= 0 && row < 28 && col >= 0 && col < 28) {
        nextGrid[row][col] = clamp(value, 0, 1);
      }
    }
  }
  return nextGrid;
}

function blurGrid(grid) {
  return grid.map((row, rowIndex) =>
    row.map((_, colIndex) => {
      let total = 0;
      let count = 0;
      for (let rowOffset = -1; rowOffset <= 1; rowOffset += 1) {
        for (let colOffset = -1; colOffset <= 1; colOffset += 1) {
          const targetRow = rowIndex + rowOffset;
          const targetCol = colIndex + colOffset;
          if (
            targetRow >= 0 &&
            targetRow < 28 &&
            targetCol >= 0 &&
            targetCol < 28
          ) {
            total += grid[targetRow][targetCol];
            count += 1;
          }
        }
      }
      return total / count;
    }),
  );
}

function buildPresetDigit(digit) {
  let grid = createEmptyGrid();
  const segments = segmentMap[digit] ?? segmentMap[8];
  const thickness = 3;

  if (segments.includes("a")) {
    grid = paintRect(grid, 3, 7, thickness, 14);
  }
  if (segments.includes("b")) {
    grid = paintRect(grid, 5, 18, 9, thickness);
  }
  if (segments.includes("c")) {
    grid = paintRect(grid, 15, 18, 9, thickness);
  }
  if (segments.includes("d")) {
    grid = paintRect(grid, 22, 7, thickness, 14);
  }
  if (segments.includes("e")) {
    grid = paintRect(grid, 15, 7, 9, thickness);
  }
  if (segments.includes("f")) {
    grid = paintRect(grid, 5, 7, 9, thickness);
  }
  if (segments.includes("g")) {
    grid = paintRect(grid, 12, 7, thickness, 14);
  }

  return blurGrid(grid);
}

function emitWithAck(socket, eventName, payload) {
  return new Promise((resolve, reject) => {
    socket.emit(eventName, payload, (response) => {
      if (response?.ok) {
        resolve(response);
        return;
      }

      reject(response?.error ?? { message: "Request failed" });
    });
  });
}

/* ── Animation variants ── */
const fadeUp = {
  hidden: { opacity: 0, y: 20 },
  visible: {
    opacity: 1,
    y: 0,
    transition: { duration: 0.55, ease: [0.16, 1, 0.3, 1] },
  },
};

const stagger = {
  hidden: { opacity: 0 },
  visible: {
    opacity: 1,
    transition: { staggerChildren: 0.07, delayChildren: 0.05 },
  },
};

const scaleIn = {
  hidden: { opacity: 0, scale: 0.92 },
  visible: {
    opacity: 1,
    scale: 1,
    transition: { duration: 0.5, ease: [0.16, 1, 0.3, 1] },
  },
};

const SHOWCASE_STEPS = [
  {
    label: "Preparing Canvas",
    description:
      "Clearing the drawing area and selecting a digit to demonstrate…",
  },
  {
    label: "Drawing Digit",
    description:
      "The network needs an input image. Watch as the digit is drawn pixel-by-pixel on the 28×28 canvas — just like handwriting.",
  },
  {
    label: "Sending to Neural Network",
    description:
      "The 784 pixel values are sent to the PyTorch backend. The neural network processes the input through its layers.",
  },
  {
    label: "Signal Propagation — Layer 1",
    description:
      "784 input neurons fire signals through weighted connections to 96 hidden neurons. Each connection strength was learned during training.",
  },
  {
    label: "Signal Propagation — Layer 2",
    description:
      "The 96 hidden neurons pass their activations through another set of learned weights to 48 neurons in the second hidden layer.",
  },
  {
    label: "Signal Propagation — Output",
    description:
      "Final layer: 48 hidden neurons connect to 10 output neurons — one for each digit (0–9). The strongest output is the prediction.",
  },
  {
    label: "Prediction Result",
    description:
      "The network has spoken! The output neuron with the highest activation determines the predicted digit and confidence.",
  },
  {
    label: "Deep Analysis",
    description:
      "Examining neuron activations, weight contributions, and confidence distribution across all 10 digit classes.",
  },
  {
    label: "3D Visualization",
    description:
      "The same network shown in 3D space — watch connections form between layers as signals flow from input to output.",
  },
  {
    label: "Showcase Complete",
    description:
      "That's how a neural network classifies handwritten digits! Every connection, activation, and prediction was computed in real time.",
  },
];

function App() {
  const [grid, setGrid] = useState(createEmptyGrid);
  const [model, setModel] = useState(null);
  const [analysis, setAnalysis] = useState(null);
  const [history, setHistory] = useState([]);
  const [layerScales, setLayerScales] = useState([1, 1, 1]);
  const [outputBiasOffsets, setOutputBiasOffsets] = useState(Array(10).fill(0));
  const [brushRadius, setBrushRadius] = useState(2);
  const [brushStrength, setBrushStrength] = useState(1);
  const [selectedBiasDigit, setSelectedBiasDigit] = useState(0);
  const [targetDigit, setTargetDigit] = useState(0);
  const [trainingLabel, setTrainingLabel] = useState(0);
  const [backpropEnabled, setBackpropEnabled] = useState(true);
  const [show3D, setShow3D] = useState(true);
  const [status, setStatus] = useState("Connecting to backend…");
  const [databaseStatus, setDatabaseStatus] = useState("History in memory");
  const [error, setError] = useState("");
  const [isPending, startTransition] = useTransition();
  const [isTraining, setIsTraining] = useState(false);
  const [socket, setSocket] = useState(null);

  const deferredAnalysis = useDeferredValue(analysis);
  const prediction = deferredAnalysis?.prediction;
  const currentBiasValue = outputBiasOffsets[selectedBiasDigit];

  // ── Showcase state ──
  const [showcaseActive, setShowcaseActive] = useState(false);
  const [showcaseStep, setShowcaseStep] = useState(-1);
  const [showcaseCaption, setShowcaseCaption] = useState("");
  const [edgeRevealProgress, setEdgeRevealProgress] = useState(null);
  const [edge3DProgress, setEdge3DProgress] = useState(null);
  const showcaseCancelRef = useRef(false);

  // ── Section refs for auto-scroll ──
  const canvasSectionRef = useRef(null);
  const networkSectionRef = useRef(null);
  const predictionSectionRef = useRef(null);
  const analysisSectionRef = useRef(null);
  const scene3DRef = useRef(null);
  const historySectionRef = useRef(null);

  useEffect(() => {
    const nextSocket = io(import.meta.env.VITE_SOCKET_URL || undefined, {
      transports: ["websocket", "polling"],
    });

    setSocket(nextSocket);

    nextSocket.on("connect", () => {
      setStatus("Realtime bridge connected");
    });

    nextSocket.on("prediction-history", (nextHistory) => {
      setHistory(nextHistory);
    });

    nextSocket.on("analysis-complete", (payload) => {
      startTransition(() => {
        setAnalysis(payload);
      });
    });

    nextSocket.on("model-refresh-required", async () => {
      try {
        const { data } = await api.get("/api/model");
        setModel(data);
      } catch (requestError) {
        setError(requestError.message);
      }
    });

    return () => {
      nextSocket.close();
    };
  }, [startTransition]);

  useEffect(() => {
    async function hydrate() {
      try {
        const [
          { data: healthData },
          { data: modelData },
          { data: historyData },
        ] = await Promise.all([
          api.get("/api/health"),
          api.get("/api/model"),
          api.get("/api/history"),
        ]);
        setStatus(`Backend ready • ${healthData.python.status}`);
        setDatabaseStatus(
          healthData.database?.connected
            ? `Atlas • ${healthData.database.databaseName}`
            : healthData.database?.message || "History in memory",
        );
        setModel(modelData);
        setHistory(historyData.history);
      } catch (requestError) {
        setError(requestError.message);
        setStatus("Backend unavailable");
        setDatabaseStatus("History unavailable");
      }
    }

    hydrate();
  }, []);

  async function analyzeDigit() {
    if (!socket) {
      return;
    }

    setError("");

    try {
      const response = await emitWithAck(socket, "analyze-digit", {
        pixels: grid,
        adjustments: {
          layerScales,
          outputBiasOffsets,
          targetDigit,
        },
        includeBackprop: backpropEnabled,
      });
      startTransition(() => {
        setAnalysis(response.analysis);
      });
    } catch (requestError) {
      setError(requestError.message ?? "Analysis failed");
    }
  }

  async function trainStep() {
    if (!socket) {
      return;
    }

    setError("");
    setIsTraining(true);

    try {
      const response = await emitWithAck(socket, "train-digit", {
        pixels: grid,
        adjustments: {
          layerScales,
          outputBiasOffsets,
          targetDigit,
        },
        includeBackprop: true,
        label: Number(trainingLabel),
      });
      startTransition(() => {
        setAnalysis(response.analysis);
      });
      const { data } = await api.get("/api/model");
      setModel(data);
    } catch (requestError) {
      setError(requestError.message ?? "Training step failed");
    } finally {
      setIsTraining(false);
    }
  }

  function updateLayerScale(index, value) {
    const nextScales = [...layerScales];
    nextScales[index] = Number(value);
    setLayerScales(nextScales);
  }

  function updateOutputBias(value) {
    const nextBiasOffsets = [...outputBiasOffsets];
    nextBiasOffsets[selectedBiasDigit] = Number(value);
    setOutputBiasOffsets(nextBiasOffsets);
  }

  // ── Showcase helpers ──
  function sleep(ms) {
    return new Promise((resolve) => setTimeout(resolve, ms));
  }

  function scrollToRef(ref) {
    ref.current?.scrollIntoView({ behavior: "smooth", block: "center" });
  }

  const stopShowcase = useCallback(() => {
    showcaseCancelRef.current = true;
    setShowcaseActive(false);
    setShowcaseStep(-1);
    setShowcaseCaption("");
    setEdgeRevealProgress(null);
    setEdge3DProgress(null);
  }, []);

  const runShowcase = useCallback(async () => {
    if (showcaseActive) {
      stopShowcase();
      return;
    }

    showcaseCancelRef.current = false;
    setShowcaseActive(true);
    setEdgeRevealProgress(null);
    setEdge3DProgress(null);

    const cancelled = () => showcaseCancelRef.current;

    function setStep(index) {
      if (cancelled()) return false;
      setShowcaseStep(index);
      setShowcaseCaption(SHOWCASE_STEPS[index]?.description ?? "");
      return true;
    }

    try {
      // Step 0: Prepare canvas
      if (!setStep(0)) return;
      setGrid(createEmptyGrid());
      setAnalysis(null);
      scrollToRef(canvasSectionRef);
      await sleep(1200);

      // Step 1: Auto-draw a random digit
      if (!setStep(1)) return;
      const digit = Math.floor(Math.random() * 10);
      const targetGrid = buildPresetDigit(digit);

      // Collect non-zero pixels and animate drawing them
      const pixels = [];
      for (let r = 0; r < 28; r++) {
        for (let c = 0; c < 28; c++) {
          if (targetGrid[r][c] > 0.05) {
            pixels.push({ r, c, v: targetGrid[r][c] });
          }
        }
      }
      // Sort roughly by stroke order (top-to-bottom, left-to-right)
      pixels.sort((a, b) => a.r - b.r || a.c - b.c);

      // Draw pixels in batches for smooth animation
      const batchSize = Math.max(2, Math.ceil(pixels.length / 40));
      let currentGrid = createEmptyGrid();
      for (let i = 0; i < pixels.length; i += batchSize) {
        if (cancelled()) return;
        const batch = pixels.slice(i, i + batchSize);
        currentGrid = currentGrid.map((row) => [...row]);
        for (const px of batch) {
          currentGrid[px.r][px.c] = px.v;
        }
        setGrid(currentGrid);
        await sleep(50);
      }
      // Final grid with blur
      setGrid(targetGrid);
      await sleep(600);

      // Step 2: Analyze
      if (!setStep(2)) return;
      scrollToRef(predictionSectionRef);
      if (socket) {
        try {
          const response = await emitWithAck(socket, "analyze-digit", {
            pixels: targetGrid,
            adjustments: { layerScales, outputBiasOffsets, targetDigit },
            includeBackprop: backpropEnabled,
          });
          startTransition(() => {
            setAnalysis(response.analysis);
          });
        } catch {
          // If analysis fails, continue showcase with whatever we have
        }
      }
      await sleep(1500);

      // Steps 3-5: Progressive edge reveal in network graph
      if (!setStep(3)) return;
      scrollToRef(networkSectionRef);
      await sleep(800);

      // Animate edge reveal from 0 to 1 over ~6 seconds (3 layers, ~2s each)
      const edgeSteps = 100;
      const layerStepMap = [
        { stepIndex: 3, start: 0, end: 0.33 },
        { stepIndex: 4, start: 0.33, end: 0.66 },
        { stepIndex: 5, start: 0.66, end: 1.0 },
      ];

      for (let s = 0; s <= edgeSteps; s++) {
        if (cancelled()) return;
        const progress = s / edgeSteps;
        setEdgeRevealProgress(progress);

        // Update step label when layer changes
        for (const layer of layerStepMap) {
          if (progress >= layer.start && progress < layer.end + 0.01) {
            setStep(layer.stepIndex);
            break;
          }
        }

        await sleep(60);
      }
      setEdgeRevealProgress(1);
      await sleep(1000);

      // Step 6: Prediction result
      if (!setStep(6)) return;
      scrollToRef(predictionSectionRef);
      await sleep(2500);

      // Step 7: Deep analysis
      if (!setStep(7)) return;
      scrollToRef(analysisSectionRef);
      await sleep(3000);

      // Step 8: 3D visualization
      if (!setStep(8)) return;
      setShow3D(true);
      await sleep(400);
      scrollToRef(scene3DRef);
      await sleep(800);

      // Animate 3D edges
      const edge3DSteps = 60;
      for (let s = 0; s <= edge3DSteps; s++) {
        if (cancelled()) return;
        setEdge3DProgress(s / edge3DSteps);
        await sleep(80);
      }
      setEdge3DProgress(1);
      await sleep(2000);

      // Step 9: Complete
      if (!setStep(9)) return;
      await sleep(4000);
    } finally {
      if (!showcaseCancelRef.current) {
        setShowcaseActive(false);
        setShowcaseStep(-1);
        setShowcaseCaption("");
      }
      setEdgeRevealProgress(null);
      setEdge3DProgress(null);
    }
  }, [
    showcaseActive,
    stopShowcase,
    socket,
    layerScales,
    outputBiasOffsets,
    targetDigit,
    backpropEnabled,
    startTransition,
  ]);

  return (
    <main className="app-shell min-h-screen text-white">
      {/* Background orbs */}
      <div className="bg-orb bg-orb-1" />
      <div className="bg-orb bg-orb-2" />
      <div className="bg-orb bg-orb-3" />

      {/* ═══ SHOWCASE OVERLAY ═══ */}
      <AnimatePresence>
        {showcaseActive && showcaseStep >= 0 && (
          <motion.div
            className="showcase-panel"
            initial={{ opacity: 0, y: 40 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: 40 }}
            transition={{ duration: 0.4, ease: [0.16, 1, 0.3, 1] }}
          >
            <div className="showcase-panel-inner">
              <div className="showcase-step-badge">
                Step {showcaseStep + 1} / {SHOWCASE_STEPS.length}
              </div>
              <h3 className="showcase-step-title">
                {SHOWCASE_STEPS[showcaseStep]?.label}
              </h3>
              <p className="showcase-step-desc">{showcaseCaption}</p>
              <div className="showcase-progress-track">
                <div
                  className="showcase-progress-fill"
                  style={{
                    width: `${((showcaseStep + 1) / SHOWCASE_STEPS.length) * 100}%`,
                  }}
                />
              </div>
              <button className="showcase-stop-btn" onClick={stopShowcase}>
                <Square size={12} />
                Stop Showcase
              </button>
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      <div className="relative z-10 mx-auto flex w-full max-w-[1600px] flex-col gap-12 px-4 py-10 sm:px-6 lg:px-8">
        {/* ═══ HERO ═══ */}
        <motion.header
          className="hero-section glass-panel p-7 sm:p-9 lg:p-11"
          initial="hidden"
          animate="visible"
          variants={stagger}
        >
          <motion.div variants={fadeUp} className="mb-5">
            <span className="section-label">
              <Sparkles size={11} />
              Live Neural Network Showcase
            </span>
          </motion.div>

          <motion.h1
            variants={fadeUp}
            className="gradient-text mb-3 max-w-4xl text-3xl font-extrabold tracking-tight sm:text-4xl lg:text-5xl xl:text-6xl"
          >
            Watch a neural network think in real time.
          </motion.h1>

          <motion.p
            variants={fadeUp}
            className="mb-8 max-w-2xl text-sm leading-relaxed text-white/45 sm:text-base"
          >
            Draw any digit on the canvas below, or press{" "}
            <strong className="text-white/70">Start Showcase</strong> for a
            fully automated step-by-step demonstration of how a neural network
            classifies handwritten digits.
          </motion.p>

          {/* Showcase CTA */}
          <motion.div variants={fadeUp} className="mb-8">
            <button
              className={`showcase-btn ${showcaseActive ? "showcase-btn-active" : ""}`}
              onClick={runShowcase}
            >
              {showcaseActive ? (
                <>
                  <Square size={16} />
                  Stop Showcase
                </>
              ) : (
                <>
                  <Play size={16} />
                  Start Automated Showcase
                  <ChevronRight size={14} className="ml-1" />
                </>
              )}
            </button>
          </motion.div>

          {/* Stats ribbon */}
          <motion.div
            variants={stagger}
            className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4"
          >
            {[
              {
                icon: Brain,
                label: "Architecture",
                value: model ? model.architecture.join(" → ") : "Loading…",
                color: "#06d6a0",
              },
              {
                icon: Activity,
                label: "Status",
                value: status,
                dot: true,
                color: "#3b82f6",
              },
              {
                icon: Layers,
                label: "Database",
                value: databaseStatus,
                color: "#7c3aed",
              },
              {
                icon: Zap,
                label: "Prediction",
                value: prediction
                  ? `${prediction.digit} — ${(prediction.confidence * 100).toFixed(1)}%`
                  : "Awaiting input",
                color: "#f59e0b",
              },
            ].map((item) => (
              <motion.div
                key={item.label}
                variants={fadeUp}
                className="stat-card"
              >
                <div className="mb-2.5 flex items-center gap-2">
                  <item.icon size={15} style={{ color: item.color }} />
                  <span className="text-[0.65rem] font-semibold uppercase tracking-[0.12em] text-white/30">
                    {item.label}
                  </span>
                  {item.dot && (
                    <div
                      className={`status-dot ml-auto ${status.includes("ready") || status.includes("connected") ? "connected" : "disconnected"}`}
                    />
                  )}
                </div>
                <p className="text-sm font-medium text-white/75 mono">
                  {item.value}
                </p>
              </motion.div>
            ))}
          </motion.div>
        </motion.header>

        {/* ═══ MAIN DEMO ═══ */}
        <motion.section
          className="grid gap-8 xl:grid-cols-[0.85fr_1.15fr]"
          initial="hidden"
          whileInView="visible"
          viewport={{ once: true, margin: "-40px" }}
          variants={stagger}
        >
          {/* Left column: Canvas + Actions + Controls */}
          <div className="flex flex-col gap-8">
            <motion.div variants={fadeUp} ref={canvasSectionRef}>
              <DigitCanvas
                grid={grid}
                onChange={setGrid}
                brushRadius={brushRadius}
                brushStrength={brushStrength}
              />
            </motion.div>

            {/* Quick actions */}
            <motion.div variants={fadeUp} className="glass-panel p-5">
              <div className="mb-4 flex items-center gap-2">
                <span className="section-label">
                  <Zap size={11} />
                  Actions
                </span>
              </div>

              <div className="flex flex-wrap gap-2">
                <button
                  className="btn-primary"
                  onClick={analyzeDigit}
                  disabled={isPending}
                >
                  <Eye size={14} />
                  {isPending ? "Analyzing…" : "Analyze"}
                </button>
                <button
                  className="btn-secondary"
                  onClick={() => setGrid(createEmptyGrid())}
                >
                  <RotateCcw size={14} />
                  Clear
                </button>
                {Array.from({ length: 10 }, (_, digit) => (
                  <button
                    key={digit}
                    className="digit-chip"
                    onClick={() => setGrid(buildPresetDigit(digit))}
                  >
                    {digit}
                  </button>
                ))}
              </div>
            </motion.div>

            {/* Controls */}
            <motion.div variants={fadeUp} className="glass-panel p-5 space-y-5">
              <div className="flex items-center gap-2">
                <span className="section-label">
                  <Settings2 size={11} />
                  Controls
                </span>
              </div>

              <div className="grid gap-4 md:grid-cols-2">
                {/* Brush */}
                <div className="inner-card p-4">
                  <p className="text-xs font-semibold uppercase tracking-[0.1em] text-white/50 mb-3">
                    Brush
                  </p>
                  <div className="space-y-3 text-sm text-white/55">
                    <label className="block">
                      <span className="mb-1.5 flex justify-between text-xs">
                        <span>Radius</span>
                        <span className="mono text-white/70">
                          {brushRadius}
                        </span>
                      </span>
                      <input
                        type="range"
                        min="1"
                        max="4"
                        step="1"
                        value={brushRadius}
                        onChange={(e) => setBrushRadius(Number(e.target.value))}
                      />
                    </label>
                    <label className="block">
                      <span className="mb-1.5 flex justify-between text-xs">
                        <span>Strength</span>
                        <span className="mono text-white/70">
                          {brushStrength.toFixed(2)}
                        </span>
                      </span>
                      <input
                        type="range"
                        min="0.4"
                        max="1.5"
                        step="0.05"
                        value={brushStrength}
                        onChange={(e) =>
                          setBrushStrength(Number(e.target.value))
                        }
                      />
                    </label>
                  </div>
                </div>

                {/* Layer scaling */}
                <div className="inner-card p-4">
                  <p className="text-xs font-semibold uppercase tracking-[0.1em] text-white/50 mb-3">
                    Layer Scaling
                  </p>
                  <div className="space-y-3 text-sm text-white/55">
                    {layerScales.map((scale, index) => (
                      <label key={index} className="block">
                        <span className="mb-1.5 flex justify-between text-xs">
                          <span>Layer {index + 1}</span>
                          <span className="mono text-white/70">
                            {scale.toFixed(2)}x
                          </span>
                        </span>
                        <input
                          type="range"
                          min="0.45"
                          max="1.75"
                          step="0.05"
                          value={scale}
                          onChange={(e) =>
                            updateLayerScale(index, e.target.value)
                          }
                        />
                      </label>
                    ))}
                  </div>
                </div>
              </div>

              <div className="grid gap-4 md:grid-cols-2">
                {/* Bias override */}
                <div className="inner-card p-4">
                  <p className="text-xs font-semibold uppercase tracking-[0.1em] text-white/50 mb-3">
                    Bias Override
                  </p>
                  <div className="space-y-3 text-sm text-white/55">
                    <label className="block">
                      <span className="mb-1.5 flex justify-between text-xs">
                        <span>Digit Channel</span>
                        <span className="mono text-white/70">
                          {selectedBiasDigit}
                        </span>
                      </span>
                      <input
                        type="range"
                        min="0"
                        max="9"
                        step="1"
                        value={selectedBiasDigit}
                        onChange={(e) =>
                          setSelectedBiasDigit(Number(e.target.value))
                        }
                      />
                    </label>
                    <label className="block">
                      <span className="mb-1.5 flex justify-between text-xs">
                        <span>Value</span>
                        <span className="mono text-white/70">
                          {currentBiasValue.toFixed(2)}
                        </span>
                      </span>
                      <input
                        type="range"
                        min="-2"
                        max="2"
                        step="0.05"
                        value={currentBiasValue}
                        onChange={(e) => updateOutputBias(e.target.value)}
                      />
                    </label>
                  </div>
                </div>

                {/* Backprop settings */}
                <div className="inner-card p-4">
                  <p className="text-xs font-semibold uppercase tracking-[0.1em] text-white/50 mb-3">
                    Backprop
                  </p>
                  <div className="space-y-3 text-sm">
                    <label className="flex items-center justify-between gap-3 rounded-xl border border-white/5 bg-black/15 px-3 py-2.5 text-xs text-white/55">
                      <span>Gradients</span>
                      <input
                        type="checkbox"
                        checked={backpropEnabled}
                        onChange={(e) => setBackpropEnabled(e.target.checked)}
                        className="h-3.5 w-3.5 accent-[#06d6a0]"
                      />
                    </label>
                    <label className="flex items-center justify-between gap-3 rounded-xl border border-white/5 bg-black/15 px-3 py-2.5 text-xs text-white/55">
                      <span>3D Scene</span>
                      <input
                        type="checkbox"
                        checked={show3D}
                        onChange={(e) => setShow3D(e.target.checked)}
                        className="h-3.5 w-3.5 accent-[#06d6a0]"
                      />
                    </label>
                    <label className="block">
                      <span className="mb-1.5 flex justify-between text-xs text-white/55">
                        <span>Target digit</span>
                        <span className="mono text-white/70">
                          {targetDigit}
                        </span>
                      </span>
                      <input
                        type="range"
                        min="0"
                        max="9"
                        step="1"
                        value={targetDigit}
                        onChange={(e) => setTargetDigit(Number(e.target.value))}
                      />
                    </label>
                  </div>
                </div>
              </div>

              {/* Training */}
              <div className="inner-card p-4">
                <div className="flex flex-wrap items-center gap-3">
                  <FlaskConical size={14} className="text-[#7c3aed]" />
                  <span className="text-xs font-semibold uppercase tracking-[0.1em] text-white/50">
                    Live Training
                  </span>
                  <div className="flex items-center gap-2 ml-auto">
                    <label className="text-xs text-white/50">Label</label>
                    <input
                      type="number"
                      min="0"
                      max="9"
                      value={trainingLabel}
                      onChange={(e) =>
                        setTrainingLabel(
                          clamp(Number(e.target.value || 0), 0, 9),
                        )
                      }
                      className="w-14 rounded-lg border border-white/8 bg-black/25 px-2 py-1.5 text-sm text-white mono"
                    />
                    <button
                      className="btn-secondary"
                      onClick={trainStep}
                      disabled={isTraining}
                    >
                      <Target size={13} />
                      {isTraining ? "Training…" : "Apply step"}
                    </button>
                  </div>
                </div>
              </div>

              {/* Error */}
              <AnimatePresence>
                {error && (
                  <motion.div
                    initial={{ opacity: 0, height: 0 }}
                    animate={{ opacity: 1, height: "auto" }}
                    exit={{ opacity: 0, height: 0 }}
                    className="rounded-xl border border-red-400/20 bg-red-500/8 px-4 py-3 text-sm text-red-200"
                  >
                    {error}
                  </motion.div>
                )}
              </AnimatePresence>
            </motion.div>
          </div>

          {/* Right column: Prediction + Network */}
          <div className="flex flex-col gap-8">
            {/* Prediction display */}
            <motion.div
              ref={predictionSectionRef}
              variants={scaleIn}
              className="glass-panel glow-border p-6 sm:p-8"
            >
              <div className="flex items-center gap-2 mb-6">
                <span className="section-label">
                  <Zap size={11} />
                  Prediction
                </span>
              </div>

              <div className="flex flex-col items-center gap-6 sm:flex-row sm:items-start">
                {/* Orb */}
                <AnimatePresence mode="wait">
                  <motion.div
                    key={prediction?.digit ?? "empty"}
                    initial={{ scale: 0.8, opacity: 0 }}
                    animate={{ scale: 1, opacity: 1 }}
                    exit={{ scale: 0.8, opacity: 0 }}
                    transition={{
                      type: "spring",
                      stiffness: 260,
                      damping: 20,
                    }}
                    className="prediction-orb flex h-28 w-28 flex-shrink-0 items-center justify-center rounded-2xl text-5xl font-bold text-white"
                  >
                    {prediction?.digit ?? "–"}
                  </motion.div>
                </AnimatePresence>

                <div className="flex flex-col gap-3 w-full">
                  <div>
                    <p className="text-[0.65rem] font-semibold uppercase tracking-[0.12em] text-white/30 mb-1">
                      Confidence
                    </p>
                    <p className="text-3xl font-bold text-white mono">
                      {prediction
                        ? `${(prediction.confidence * 100).toFixed(1)}%`
                        : "—"}
                    </p>
                  </div>
                  <p className="text-xs leading-relaxed text-white/40">
                    The network highlights its dominant signal paths. Strongest
                    contributing hidden neurons are surfaced in the analysis
                    below.
                  </p>
                </div>
              </div>

              {/* Backprop metrics */}
              {deferredAnalysis?.backprop && (
                <motion.div
                  initial={{ opacity: 0, y: 10 }}
                  animate={{ opacity: 1, y: 0 }}
                  className="mt-6 grid gap-3 sm:grid-cols-2 lg:grid-cols-4"
                >
                  <div className="inner-card p-3.5">
                    <p className="text-[0.6rem] uppercase tracking-[0.1em] text-white/35">
                      Loss
                    </p>
                    <p className="mt-1 text-lg font-bold text-white mono">
                      {deferredAnalysis.backprop.loss.toFixed(4)}
                    </p>
                  </div>
                  <div className="inner-card p-3.5">
                    <p className="text-[0.6rem] uppercase tracking-[0.1em] text-white/35">
                      Target
                    </p>
                    <p className="mt-1 text-lg font-bold text-white mono">
                      {deferredAnalysis.backprop.targetDigit}
                    </p>
                  </div>
                  {deferredAnalysis.backprop.layerGradientNorms.map(
                    (norm, i) => (
                      <div key={i} className="inner-card p-3.5">
                        <p className="text-[0.6rem] uppercase tracking-[0.1em] text-white/35">
                          ∇ Layer {i + 1}
                        </p>
                        <p className="mt-1 text-lg font-bold text-white mono">
                          {norm.toFixed(4)}
                        </p>
                      </div>
                    ),
                  )}
                </motion.div>
              )}
            </motion.div>

            {/* Network graph */}
            <motion.div variants={fadeUp} ref={networkSectionRef}>
              <NetworkGraph
                model={model}
                analysis={deferredAnalysis}
                edgeRevealProgress={edgeRevealProgress}
              />
            </motion.div>
          </div>
        </motion.section>

        {/* ═══ ACTIVATION ANALYSIS ═══ */}
        <motion.div
          ref={analysisSectionRef}
          className="mt-4"
          initial="hidden"
          whileInView="visible"
          viewport={{ once: true, margin: "-40px" }}
          variants={fadeUp}
        >
          <ActivationCharts model={model} analysis={deferredAnalysis} />
        </motion.div>

        {/* ═══ 3D SCENE ═══ */}
        <AnimatePresence>
          {show3D && (
            <motion.div
              ref={scene3DRef}
              className="mt-4"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -10 }}
              transition={{ duration: 0.4 }}
            >
              <Suspense
                fallback={
                  <div className="nn-viz-container flex min-h-[240px] items-center justify-center text-sm text-white/40 shimmer">
                    <Box size={18} className="mr-2 animate-spin" />
                    Loading 3D network scene…
                  </div>
                }
              >
                <NetworkScene3D
                  analysis={deferredAnalysis}
                  edgeRevealProgress={edge3DProgress}
                />
              </Suspense>
            </motion.div>
          )}
        </AnimatePresence>

        {/* ═══ HISTORY ═══ */}
        <motion.div
          ref={historySectionRef}
          className="mt-4 mb-8"
          initial="hidden"
          whileInView="visible"
          viewport={{ once: true, margin: "-40px" }}
          variants={fadeUp}
        >
          <PredictionHistory history={history} />
        </motion.div>
      </div>
    </main>
  );
}

export default App;
