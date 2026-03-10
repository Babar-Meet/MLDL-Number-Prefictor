import { useState, useEffect, useRef, useMemo, useCallback } from "react";
import { Canvas, useFrame, useThree } from "@react-three/fiber";
import { OrbitControls, Text, Html } from "@react-three/drei";
import * as THREE from "three";
import { Play, Pause, SkipBack, SkipForward, RotateCcw } from "lucide-react";

/*
 * Step 0 — Input Matrix: 28×28 grid with pixel-value numbers
 * Step 1 — Matrix × Weights: animated math breakdown
 * Step 2 — Hidden Layers: data particles flow into bars
 * Step 3 — Activation & Output: final prediction highlight
 */

const TOTAL_STEPS = 5;
const STEP_DURATION_MS = 4500;
const LERP_SPEED = 2.5;

function clamp(v, lo, hi) {
  return Math.max(lo, Math.min(hi, v));
}

/* ── Camera targets per step ── */
const CAMERA_TARGETS = [
  { pos: [0, 0, 14], look: [0, 0, 0] },
  { pos: [12, 1.5, 4], look: [-2, 0, 0] },
  { pos: [4, 1, 13], look: [1, 0, 0] },
  { pos: [-1, 3, 18], look: [0, 0, -2] },
  { pos: [0, -2, 16], look: [0, -3, 0] },
];

function CameraRig({ step }) {
  const { camera } = useThree();
  const target = useRef(new THREE.Vector3());
  const posTarget = useRef(new THREE.Vector3());

  useEffect(() => {
    const t = CAMERA_TARGETS[step] ?? CAMERA_TARGETS[0];
    posTarget.current.set(...t.pos);
    target.current.set(...t.look);
  }, [step]);

  useFrame((_, delta) => {
    const speed = LERP_SPEED * delta;
    camera.position.lerp(posTarget.current, speed);
    const lookAt = new THREE.Vector3().copy(camera.position);
    lookAt.lerp(target.current, 0.5);
    camera.lookAt(target.current);
  });

  return null;
}

/* ── 28×28 cubes with floating value labels ── */
function InputMatrix({ grid, position, visible, stepProgress }) {
  const groupRef = useRef();
  const meshRef = useRef();
  const dummy = useMemo(() => new THREE.Object3D(), []);

  // Build a set of text labels for non-zero pixels (sampled — every 4th row/col)
  const labels = useMemo(() => {
    if (!grid) return [];
    const out = [];
    const sp = 0.22;
    for (let r = 0; r < 28; r += 4) {
      for (let c = 0; c < 28; c += 4) {
        const v = grid[r][c];
        if (v > 0.05) {
          out.push({
            key: `${r}-${c}`,
            x: c * sp - 14 * sp,
            y: (28 - r) * sp - 14 * sp,
            z: v * 0.5 + 0.25,
            label: v.toFixed(1),
            val: v,
          });
        }
      }
    }
    return out;
  }, [grid]);

  useFrame((state) => {
    if (!visible || !grid || !meshRef.current) return;
    const t = state.clock.elapsedTime;
    let i = 0;
    const sp = 0.22;
    for (let r = 0; r < 28; r++) {
      for (let c = 0; c < 28; c++) {
        const val = grid[r][c] || 0;
        // Wave entrance animation
        const delay = (r + c) * 0.012;
        const anim = clamp((stepProgress - delay) * 3, 0, 1);
        const height = Math.max(0.05, val * 2.5 * anim);
        dummy.position.set(
          c * sp - 14 * sp,
          (28 - r) * sp - 14 * sp,
          height * 0.25,
        );
        dummy.scale.set(anim, anim, height);
        const color = new THREE.Color();
        const hue = 0.55 + val * 0.12;
        color.setHSL(hue, 0.85, 0.15 + val * 0.65);
        dummy.updateMatrix();
        meshRef.current.setMatrixAt(i, dummy.matrix);
        meshRef.current.setColorAt(i, color);
        i++;
      }
    }
    meshRef.current.instanceMatrix.needsUpdate = true;
    meshRef.current.instanceColor.needsUpdate = true;

    // Gentle float
    if (groupRef.current) {
      groupRef.current.position.y = position[1] + Math.sin(t * 0.8) * 0.08;
    }
  });

  return (
    <group ref={groupRef} position={position} visible={visible}>
      <Text
        position={[0, 4, 0]}
        fontSize={0.4}
        color="#00ff9d"
        anchorX="center"
      >
        INPUT MATRIX · 28 × 28 = 784 pixels
      </Text>

      <instancedMesh ref={meshRef} args={[null, null, 28 * 28]}>
        <boxGeometry args={[0.18, 0.18, 0.18]} />
        <meshStandardMaterial toneMapped={false} />
      </instancedMesh>

      {/* Floating value labels */}
      {labels.map((l) => (
        <Text
          key={l.key}
          position={[l.x, l.y, l.z]}
          fontSize={0.12}
          color={`hsl(${160 + l.val * 40}, 90%, ${50 + l.val * 30}%)`}
          anchorX="center"
          anchorY="middle"
        >
          {l.label}
        </Text>
      ))}
    </group>
  );
}

/* ── Activation bars (hidden / output layers) ── */
function ActivationBars({
  values,
  position,
  title,
  visible,
  hueBase,
  stepProgress,
  showLabels,
}) {
  const meshRef = useRef();
  const groupRef = useRef();
  const dummy = useMemo(() => new THREE.Object3D(), []);
  const count = values?.length || 1;

  useFrame((state) => {
    if (!visible || !values || !meshRef.current) return;
    const len = values.length;
    const sp = Math.min(0.35, 14 / len);
    for (let c = 0; c < len; c++) {
      const raw = clamp(values[c] || 0, 0, 1);
      const delay = c * 0.008;
      const anim = clamp((stepProgress - delay) * 2.5, 0, 1);
      const val = raw * anim;
      const h = Math.max(0.08, val * 3.5);
      dummy.position.set(c * sp - (len * sp) / 2, h / 2, 0);
      dummy.scale.set(0.8, h, 0.8);
      const color = new THREE.Color().setHSL(
        hueBase + val * 0.15,
        0.8,
        0.15 + val * 0.55,
      );
      dummy.updateMatrix();
      meshRef.current.setMatrixAt(c, dummy.matrix);
      meshRef.current.setColorAt(c, color);
    }
    meshRef.current.instanceMatrix.needsUpdate = true;
    meshRef.current.instanceColor.needsUpdate = true;

    if (groupRef.current)
      groupRef.current.position.y =
        position[1] +
        Math.sin(state.clock.elapsedTime * 0.6 + hueBase * 10) * 0.06;
  });

  // Best value for output labels
  const bestIdx = useMemo(() => {
    if (!showLabels || !values) return -1;
    let mx = -1;
    let mi = 0;
    values.forEach((v, i) => {
      if (v > mx) {
        mx = v;
        mi = i;
      }
    });
    return mi;
  }, [values, showLabels]);

  return (
    <group ref={groupRef} position={position} visible={visible}>
      <Text
        position={[0, 4.5, 0]}
        fontSize={0.35}
        color="white"
        anchorX="center"
      >
        {title}
      </Text>
      <instancedMesh ref={meshRef} args={[null, null, count]}>
        <boxGeometry args={[0.2, 1, 0.2]} />
        <meshStandardMaterial toneMapped={false} />
      </instancedMesh>

      {showLabels &&
        values &&
        values.map((v, i) => {
          const sp = Math.min(0.35, 14 / values.length);
          const isBest = i === bestIdx;
          return (
            <Text
              key={i}
              position={[i * sp - (values.length * sp) / 2, -0.5, 0]}
              fontSize={isBest ? 0.3 : 0.2}
              color={isBest ? "#00ff9d" : "#ffffff80"}
              anchorX="center"
            >
              {i}
            </Text>
          );
        })}
    </group>
  );
}

/* ── Flowing particles (data flow between layers) ── */
function makeSeed() {
  return {
    t: Math.random(),
    ox: (Math.random() - 0.5) * 2,
    oy: (Math.random() - 0.5) * 2,
    speed: 0.15 + Math.random() * 0.3,
  };
}

function DataParticles({ from, to, visible, count }) {
  const meshRef = useRef();
  const dummy = useMemo(() => new THREE.Object3D(), []);
  const seedsRef = useRef([]);
  useEffect(() => {
    seedsRef.current = Array.from({ length: count }, makeSeed);
  }, [count]);

  useFrame((state, delta) => {
    if (!visible || !meshRef.current) return;
    const color = new THREE.Color("#00ff9d");
    for (let i = 0; i < count; i++) {
      const s = seedsRef.current[i];
      s.t = (s.t + delta * s.speed) % 1;
      const t = s.t;
      dummy.position.set(
        from[0] + (to[0] - from[0]) * t + Math.sin(t * 6) * s.ox * 0.3,
        from[1] + (to[1] - from[1]) * t + Math.cos(t * 4) * s.oy * 0.3,
        from[2] + (to[2] - from[2]) * t,
      );
      const sc = 0.06 + Math.sin(t * Math.PI) * 0.06;
      dummy.scale.setScalar(sc);
      dummy.updateMatrix();
      meshRef.current.setMatrixAt(i, dummy.matrix);
      meshRef.current.setColorAt(i, color);
    }
    meshRef.current.instanceMatrix.needsUpdate = true;
    meshRef.current.instanceColor.needsUpdate = true;
  });

  return (
    <instancedMesh ref={meshRef} args={[null, null, count]} visible={visible}>
      <sphereGeometry args={[1, 6, 6]} />
      <meshStandardMaterial
        emissive="#00ff9d"
        emissiveIntensity={2}
        toneMapped={false}
      />
    </instancedMesh>
  );
}

/* ── Math breakdown overlay ── */
function MathOverlay({ grid, position, visible }) {
  if (!visible || !grid) return null;
  const px = [grid[13][13], grid[13][14], grid[14][13], grid[14][14]];
  const wt = [0.82, -0.47, 1.13, -0.31];
  const sum = px.reduce((s, p, i) => s + p * wt[i], 0);
  const relu = Math.max(0, sum);

  return (
    <group position={position} visible={visible}>
      <Text
        position={[0, 2.5, 0]}
        fontSize={0.35}
        color="#00ff9d"
        anchorX="center"
      >
        MATRIX × WEIGHTS → ReLU
      </Text>
      <Html center position={[0, 0, 0]}>
        <div
          className="pointer-events-none select-none rounded-2xl border border-[#00ff9d]/20 bg-black/80 px-5 py-4 font-mono text-[0.8rem] text-white shadow-[0_0_40px_rgba(0,255,157,0.08)] backdrop-blur"
          style={{ minWidth: 340 }}
        >
          <div className="mb-2 text-center text-xs tracking-widest text-[#00ff9d]/60">
            SINGLE NEURON COMPUTATION
          </div>
          <table className="mx-auto border-separate border-spacing-x-3 border-spacing-y-1">
            <thead>
              <tr className="text-[0.65rem] text-white/40">
                <th>Pixel</th>
                <th />
                <th>Weight</th>
                <th />
                <th>Product</th>
              </tr>
            </thead>
            <tbody>
              {px.map((p, i) => (
                <tr key={i}>
                  <td className="text-cyan-300">{p.toFixed(2)}</td>
                  <td className="text-white/30">×</td>
                  <td
                    className={wt[i] >= 0 ? "text-green-400" : "text-red-400"}
                  >
                    {wt[i] > 0 ? "+" : ""}
                    {wt[i].toFixed(2)}
                  </td>
                  <td className="text-white/30">=</td>
                  <td className="text-white">{(p * wt[i]).toFixed(3)}</td>
                </tr>
              ))}
            </tbody>
          </table>
          <div className="mt-3 flex items-center justify-center gap-3 border-t border-white/10 pt-3 text-sm">
            <span className="text-white/50">Σ =</span>
            <span className="text-white">{sum.toFixed(3)}</span>
            <span className="text-[#00ff9d]/50">→ ReLU →</span>
            <span className="text-lg font-bold text-[#00ff9d]">
              {relu.toFixed(3)}
            </span>
          </div>
        </div>
      </Html>
    </group>
  );
}

/* ── Prediction result badge ── */
function PredictionBadge({ output, position, visible }) {
  if (!visible || !output) return null;
  let best = 0;
  output.forEach((v, i) => {
    if (v > output[best]) best = i;
  });
  const conf = (output[best] * 100).toFixed(1);

  return (
    <group position={position}>
      <Html center>
        <div className="pointer-events-none select-none rounded-2xl border border-[#00ff9d]/30 bg-black/80 px-6 py-4 text-center shadow-[0_0_60px_rgba(0,255,157,0.12)] backdrop-blur">
          <div className="text-xs tracking-widest text-[#00ff9d]/60 uppercase">
            Prediction
          </div>
          <div className="mt-1 text-5xl font-black text-[#00ff9d] drop-shadow-[0_0_12px_rgba(0,255,157,0.5)]">
            {best}
          </div>
          <div className="mt-1 text-sm text-white/60">{conf}% confidence</div>
        </div>
      </Html>
    </group>
  );
}

/* ═════════════════════════  MAIN  ═════════════════════════ */

export default function NetworkPipeline3D({ analysis, autoPlay = false }) {
  const [step, setStep] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [progress, setProgress] = useState(0); // 0‑1 within current step
  const timerRef = useRef(null);
  const startTimeRef = useRef(0);

  useEffect(() => {
    startTimeRef.current = Date.now();
  }, []);

  // Auto-play when triggered externally (e.g. showcase finishes 3D network)
  const prevAutoPlay = useRef(false);
  useEffect(() => {
    if (autoPlay && !prevAutoPlay.current) {
      // Schedule state updates in a microtask to avoid sync setState in effect
      Promise.resolve().then(() => {
        setStep(0);
        setProgress(0);
        startTimeRef.current = Date.now();
        setPlaying(true);
      });
    }
    prevAutoPlay.current = autoPlay;
  }, [autoPlay]);

  const inputGrid = analysis?.layers?.[0]?.grid;
  const hidden1 = analysis?.layers?.[1]?.activations;
  const hidden2 = analysis?.layers?.[2]?.activations;
  const output = analysis?.prediction?.probabilities;
  const hasData = !!inputGrid;

  /* ── auto-advance ── */
  useEffect(() => {
    if (!playing || !hasData) return;
    startTimeRef.current = Date.now();
    const id = setInterval(() => {
      const elapsed = Date.now() - startTimeRef.current;
      const p = clamp(elapsed / STEP_DURATION_MS, 0, 1);
      setProgress(p);
      if (p >= 1) {
        setStep((prev) => {
          const next = prev + 1;
          if (next >= TOTAL_STEPS) {
            setPlaying(false);
            return prev;
          }
          startTimeRef.current = Date.now();
          setProgress(0);
          return next;
        });
      }
    }, 30);
    timerRef.current = id;
    return () => clearInterval(id);
  }, [playing, step, hasData]);

  const restart = useCallback(() => {
    setStep(0);
    setProgress(0);
    startTimeRef.current = Date.now();
    setPlaying(true);
  }, []);

  const jumpStep = useCallback(
    (dir) => {
      const next = clamp(step + dir, 0, TOTAL_STEPS - 1);
      setStep(next);
      setProgress(0);
      startTimeRef.current = Date.now();
    },
    [step],
  );

  const STEP_LABELS = [
    "Input Matrix (28×28 pixels)",
    "Side View — Pixel Heights",
    "Matrix × Weights Computation",
    "Hidden Layer Processing",
    "Activation → Prediction",
  ];

  const STEP_SUB = [
    "Each cell = one pixel value (0-1). The grid with 784 values is the network's input.",
    "Viewing the input from the side reveals how each pixel's brightness maps to height — darker pixels stay flat while brighter ones bump up.",
    "Each pixel is multiplied by a learned weight. All products are summed, then activated with ReLU.",
    "Data flows through 96 → 48 neurons. Watch the signal propagation in real time.",
    "10 output neurons compete — the strongest activation wins and determines the predicted digit.",
  ];

  return (
    <div className="nn-viz-container mt-4">
      {/* ── Header toolbar ── */}
      <div className="nn-viz-header">
        <div className="nn-viz-title">
          <span className="nn-viz-dot" />
          Pipeline Visualization
        </div>

        <div className="flex items-center gap-1.5">
          <button
            onClick={() => jumpStep(-1)}
            className="rounded p-1 text-white/60 hover:bg-white/10 hover:text-white disabled:opacity-25"
            disabled={step === 0}
          >
            <SkipBack size={14} />
          </button>
          <button
            onClick={() => setPlaying((p) => !p)}
            className="rounded p-1 text-white/60 hover:bg-white/10 hover:text-white"
          >
            {playing ? <Pause size={14} /> : <Play size={14} />}
          </button>
          <button
            onClick={() => jumpStep(1)}
            className="rounded p-1 text-white/60 hover:bg-white/10 hover:text-white disabled:opacity-25"
            disabled={step >= TOTAL_STEPS - 1}
          >
            <SkipForward size={14} />
          </button>
          <button
            onClick={restart}
            className="rounded p-1 text-white/60 hover:bg-white/10 hover:text-white"
          >
            <RotateCcw size={14} />
          </button>
        </div>
      </div>

      {/* ── Step indicator bar ── */}
      <div className="flex items-center gap-0 bg-black/30 px-3 py-1.5 border-b border-white/5">
        {STEP_LABELS.map((label, i) => (
          <button
            key={i}
            onClick={() => {
              setStep(i);
              setProgress(0);
              startTimeRef.current = Date.now();
            }}
            className={`flex-1 rounded-md px-2 py-1 text-center text-[0.6rem] font-medium transition-all ${
              i === step
                ? "bg-[#00ff9d]/15 text-[#00ff9d] shadow-[inset_0_0_8px_rgba(0,255,157,0.08)]"
                : i < step
                  ? "text-white/50"
                  : "text-white/25"
            }`}
          >
            {label}
          </button>
        ))}
      </div>

      {/* ── Caption ── */}
      <div className="bg-black/20 px-4 py-2 text-[0.7rem] text-white/50 border-b border-white/5">
        <span className="text-[#00ff9d]/70 font-semibold">Step {step + 1}</span>
        {" — "}
        {STEP_SUB[step]}
      </div>

      {/* ── 3D Canvas ── */}
      <div className="nn-3d-canvas relative">
        <Canvas
          camera={{ position: [0, 0, 14], fov: 50 }}
          style={{ width: "100%", height: "100%" }}
        >
          <color attach="background" args={["#080b12"]} />
          <fog attach="fog" args={["#080b12", 35, 60]} />
          <ambientLight intensity={0.8} />
          <pointLight position={[8, 8, 8]} intensity={60} color="#00ff9d" />
          <pointLight position={[-6, -4, 6]} intensity={30} color="#7c3aed" />
          <pointLight position={[0, -8, 4]} intensity={20} color="#3b82f6" />

          <CameraRig step={step} />

          {/* Step 0 + 1 + 2: input matrix */}
          <InputMatrix
            grid={inputGrid}
            position={[-2, 0, 0]}
            visible={step <= 2}
            stepProgress={step === 0 ? progress : 1}
          />

          {/* Step 2: math overlay */}
          <MathOverlay
            grid={inputGrid}
            position={[4.5, 0, 0]}
            visible={step === 2}
          />

          {/* Step 3: hidden layers */}
          <ActivationBars
            values={hidden1}
            position={[0, 2.5, -3]}
            title="Hidden Layer 1 — 96 neurons (ReLU)"
            visible={step >= 3}
            hueBase={0.55}
            stepProgress={step === 3 ? progress : 1}
          />
          <ActivationBars
            values={hidden2}
            position={[0, -1, -3]}
            title="Hidden Layer 2 — 48 neurons (ReLU)"
            visible={step >= 3}
            hueBase={0.75}
            stepProgress={step === 3 ? clamp(progress * 2 - 0.4, 0, 1) : 1}
          />

          {/* Step 4: output */}
          <ActivationBars
            values={output}
            position={[0, -4.5, -1]}
            title="Output Layer — 10 classes"
            visible={step >= 4}
            hueBase={0.1}
            stepProgress={step === 4 ? progress : 1}
            showLabels
          />

          <PredictionBadge
            output={output}
            position={[0, -7.5, -1]}
            visible={step >= 4 && progress > 0.5}
          />

          {/* Data-flow particles */}
          <DataParticles
            from={[-2, 0, 0]}
            to={[0, 2.5, -3]}
            visible={step === 3 && progress < 0.6}
            count={40}
          />
          <DataParticles
            from={[0, 2.5, -3]}
            to={[0, -1, -3]}
            visible={step === 3 && progress > 0.3}
            count={30}
          />
          <DataParticles
            from={[0, -1, -3]}
            to={[0, -4.5, -1]}
            visible={step === 4 && progress < 0.6}
            count={25}
          />

          <OrbitControls
            enablePan={false}
            enableZoom={true}
            minDistance={4}
            maxDistance={40}
          />
        </Canvas>

        {/* Progress bar at bottom */}
        <div className="absolute bottom-0 left-0 right-0 h-0.5 bg-white/5">
          <div
            className="h-full bg-[#00ff9d]/60 transition-[width] duration-100"
            style={{ width: `${((step + progress) / TOTAL_STEPS) * 100}%` }}
          />
        </div>

        {!hasData && (
          <div className="absolute inset-0 flex items-center justify-center text-sm text-white/40">
            Draw a digit and analyze to see the pipeline.
          </div>
        )}
      </div>
    </div>
  );
}
