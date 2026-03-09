import { useMemo } from "react";

function clamp(v, lo, hi) {
  return Math.max(lo, Math.min(hi, v));
}

/* ── 3B1B-style color helpers ── */
function nodeColor(value) {
  const t = clamp(value ?? 0, 0, 1);
  const r = Math.round(51 + t * (0 - 51));
  const g = Math.round(51 + t * (255 - 51));
  const b = Math.round(51 + t * (157 - 51));
  return `rgb(${r},${g},${b})`;
}

function edgeColor(value) {
  const mag = clamp(Math.abs(value ?? 0) * 3, 0, 1);
  if (value >= 0) {
    return `rgba(0,255,157,${0.04 + mag * 0.35})`;
  }
  return `rgba(239,68,68,${0.04 + mag * 0.3})`;
}

function edgeWidth(value) {
  return 0.4 + clamp(Math.abs(value ?? 0) * 4, 0, 2.2);
}

/* ── Position builders ── */
function buildInputPositions() {
  const positions = [];
  const startX = 80;
  const startY = 65;
  const spacing = 6.8;
  for (let row = 0; row < 28; row += 1) {
    for (let col = 0; col < 28; col += 1) {
      positions.push({ x: startX + col * spacing, y: startY + row * spacing });
    }
  }
  return positions;
}

function buildColumnPositions(size, x, startY, endY) {
  return Array.from({ length: size }, (_, i) => ({
    x,
    y: startY + (i / Math.max(size - 1, 1)) * (endY - startY),
  }));
}

/* ── Dense connection sampling ── */
function buildDenseConnections(
  sourcePositions,
  targetPositions,
  sourceValues,
  targetValues,
  maxEdges,
) {
  const edges = [];
  const sLen = sourcePositions.length;
  const tLen = targetPositions.length;
  const perTarget = Math.max(2, Math.ceil(maxEdges / tLen));

  for (let t = 0; t < tLen; t++) {
    const step = Math.max(1, Math.floor(sLen / perTarget));
    for (let s = 0; s < sLen; s += step) {
      const sourceAct = sourceValues ? (sourceValues[s] ?? 0) : 0;
      const targetAct = targetValues ? (targetValues[t] ?? 0) : 0;
      edges.push({
        sx: sourcePositions[s].x,
        sy: sourcePositions[s].y,
        tx: targetPositions[t].x,
        ty: targetPositions[t].y,
        signal: sourceAct * targetAct,
      });
    }
  }
  return edges;
}

/* ── Layout constants ── */
const VB_W = 1300;
const VB_H = 520;
const INPUT_X = 95;
const H1_X = 440;
const H2_X = 730;
const OUTPUT_X = 1050;
const TOP_Y = 65;
const BOT_Y = 470;

const PARAM_WEIGHTS = 784 * 96 + 96 * 48 + 48 * 10;
const PARAM_BIASES = 96 + 48 + 10;
const PARAM_TOTAL = PARAM_WEIGHTS + PARAM_BIASES;

const EQUATIONS = [
  { x: (INPUT_X + H1_X) / 2, label: "a¹ = σ(W¹·a⁰ + b¹)" },
  { x: (H1_X + H2_X) / 2, label: "a² = σ(W²·a¹ + b²)" },
  { x: (H2_X + OUTPUT_X) / 2, label: "ŷ = softmax(W³·a² + b³)" },
];

const LAYER_LABELS = [
  { x: INPUT_X, label: "Input Layer", count: "784 neurons", idx: -1 },
  { x: H1_X, label: "Hidden Layer 1", count: "96 neurons", idx: 0 },
  { x: H2_X, label: "Hidden Layer 2", count: "48 neurons", idx: 1 },
  { x: OUTPUT_X, label: "Output Layer", count: "10 neurons", idx: 2 },
];

export default function NetworkGraph({ model, analysis, edgeRevealProgress }) {
  const inputPositions = useMemo(() => buildInputPositions(), []);
  const hidden1Positions = useMemo(
    () => buildColumnPositions(96, H1_X, TOP_Y + 8, BOT_Y - 8),
    [],
  );
  const hidden2Positions = useMemo(
    () => buildColumnPositions(48, H2_X, TOP_Y + 30, BOT_Y - 30),
    [],
  );
  const outputPositions = useMemo(
    () => buildColumnPositions(10, OUTPUT_X, TOP_Y + 70, BOT_Y - 70),
    [],
  );

  const inputGrid =
    analysis?.layers?.[0]?.grid ??
    Array.from({ length: 28 }, () => Array(28).fill(0));
  const inputFlat = useMemo(() => inputGrid.flat(), [inputGrid]);
  const hidden1 = analysis?.layers?.[1]?.activations ?? Array(96).fill(0);
  const hidden2 = analysis?.layers?.[2]?.activations ?? Array(48).fill(0);
  const output = analysis?.prediction?.probabilities ?? Array(10).fill(0);
  const predicted = output.indexOf(Math.max(...output));

  /* ── Dense edge webs ── */
  const denseEdgesL0 = useMemo(
    () =>
      buildDenseConnections(
        inputPositions,
        hidden1Positions,
        inputFlat,
        hidden1,
        380,
      ),
    [inputPositions, hidden1Positions, inputFlat, hidden1],
  );
  const denseEdgesL1 = useMemo(
    () =>
      buildDenseConnections(
        hidden1Positions,
        hidden2Positions,
        hidden1,
        hidden2,
        320,
      ),
    [hidden1Positions, hidden2Positions, hidden1, hidden2],
  );
  const denseEdgesL2 = useMemo(
    () =>
      buildDenseConnections(
        hidden2Positions,
        outputPositions,
        hidden2,
        output,
        200,
      ),
    [hidden2Positions, outputPositions, hidden2, output],
  );

  /* ── Dynamic edges from analysis ── */
  const dynamicEdges = analysis
    ? [
        ...(analysis.dynamicEdges?.input_to_hidden_1 ?? []).map((e) => ({
          ...e,
          layer: 0,
          value: e.signal,
        })),
        ...(analysis.dynamicEdges?.hidden_1_to_hidden_2 ?? []).map((e) => ({
          ...e,
          layer: 1,
          value: e.signal,
        })),
        ...(analysis.dynamicEdges?.hidden_2_to_output ?? []).map((e) => ({
          ...e,
          layer: 2,
          value: e.signal,
        })),
      ]
    : [];

  const layerPositions = [
    [inputPositions, hidden1Positions],
    [hidden1Positions, hidden2Positions],
    [hidden2Positions, outputPositions],
  ];

  /* ── Showcase reveal helpers ── */
  const isShowcase = edgeRevealProgress != null;

  function getLayerProgress(layerIndex) {
    if (!isShowcase) return 1;
    const layerStart = layerIndex / 3;
    const layerEnd = (layerIndex + 1) / 3;
    return clamp(
      (edgeRevealProgress - layerStart) / (layerEnd - layerStart),
      0,
      1,
    );
  }

  function getNodeOpacity(layerIdx) {
    if (!isShowcase) return 1;
    if (layerIdx < 0) return edgeRevealProgress > 0 ? 1 : 0.3;
    return getLayerProgress(layerIdx) > 0 ? 1 : 0.3;
  }

  function getDenseOpacity(layerIdx) {
    if (!isShowcase) return 1;
    return getLayerProgress(layerIdx);
  }

  return (
    <div className="nn-viz-container">
      {/* Header */}
      <div className="nn-viz-header">
        <div className="nn-viz-title">
          <span className="nn-viz-dot" />
          Neural Network Architecture
        </div>
        <div className="nn-viz-meta">
          {isShowcase && (
            <span className="showcase-layer-badge">
              {edgeRevealProgress < 0.33
                ? "Layer 1 — Input → Hidden₁"
                : edgeRevealProgress < 0.66
                  ? "Layer 2 — Hidden₁ → Hidden₂"
                  : "Layer 3 — Hidden₂ → Output"}
            </span>
          )}
          <span className="nn-arch-tag">
            {model ? model.architecture.join(" → ") : "784 → 96 → 48 → 10"}
          </span>
        </div>
      </div>

      {/* Parameter count (3B1B yellow) */}
      <div className="nn-param-panel">
        <span className="nn-param-label">Parameters</span>
        <span className="nn-param-value">
          <span className="nn-param-highlight">
            {PARAM_TOTAL.toLocaleString()}
          </span>
          <span className="nn-param-detail">
            Weights: {(784 * 96).toLocaleString()} +{" "}
            {(96 * 48).toLocaleString()} + {(48 * 10).toLocaleString()} ={" "}
            {PARAM_WEIGHTS.toLocaleString()}
            {" · "}Biases: {PARAM_BIASES}
          </span>
        </span>
      </div>

      {/* SVG canvas */}
      <div className="nn-viz-canvas">
        <svg viewBox={`0 0 ${VB_W} ${VB_H}`} className="nn-viz-svg">
          <defs>
            <filter id="nn-glow">
              <feGaussianBlur stdDeviation="3" result="blur" />
              <feComposite in="SourceGraphic" in2="blur" operator="over" />
            </filter>
            <filter id="nn-glow-strong">
              <feGaussianBlur stdDeviation="5" result="blur" />
              <feComposite in="SourceGraphic" in2="blur" operator="over" />
            </filter>
            <filter id="nn-edge-glow">
              <feGaussianBlur stdDeviation="2.5" result="blur" />
              <feComposite in="SourceGraphic" in2="blur" operator="over" />
            </filter>
          </defs>

          {/* Background */}
          <rect x="0" y="0" width={VB_W} height={VB_H} rx="12" fill="#0a0a0a" />

          {/* ═══ Dense connection webs ═══ */}
          <g opacity={getDenseOpacity(0) * 0.7}>
            {denseEdgesL0.map((e, i) => (
              <line
                key={`d0-${i}`}
                x1={e.sx}
                y1={e.sy}
                x2={e.tx}
                y2={e.ty}
                stroke={edgeColor(e.signal)}
                strokeWidth={edgeWidth(e.signal) * 0.5}
              />
            ))}
          </g>

          <g opacity={getDenseOpacity(1) * 0.7}>
            {denseEdgesL1.map((e, i) => (
              <line
                key={`d1-${i}`}
                x1={e.sx}
                y1={e.sy}
                x2={e.tx}
                y2={e.ty}
                stroke={edgeColor(e.signal)}
                strokeWidth={edgeWidth(e.signal) * 0.6}
              />
            ))}
          </g>

          <g opacity={getDenseOpacity(2) * 0.8}>
            {denseEdgesL2.map((e, i) => (
              <line
                key={`d2-${i}`}
                x1={e.sx}
                y1={e.sy}
                x2={e.tx}
                y2={e.ty}
                stroke={edgeColor(e.signal)}
                strokeWidth={edgeWidth(e.signal) * 0.7}
              />
            ))}
          </g>

          {/* ═══ Dynamic edges (strong signal paths) ═══ */}
          {dynamicEdges.map((edge, index) => {
            const [srcPos, tgtPos] = layerPositions[edge.layer];
            const source = srcPos[edge.from];
            const target = tgtPos[edge.to];
            if (!source || !target) return null;

            const lineLen = Math.hypot(
              target.x - source.x,
              target.y - source.y,
            );
            const layerProg = getLayerProgress(edge.layer);
            if (isShowcase && layerProg <= 0) return null;

            return (
              <line
                key={`dyn-${edge.layer}-${index}`}
                className="network-edge"
                x1={source.x}
                y1={source.y}
                x2={target.x}
                y2={target.y}
                stroke={edgeColor(edge.value)}
                strokeWidth={1.2 + clamp((edge.magnitude ?? 0) * 14, 0, 3.5)}
                filter="url(#nn-edge-glow)"
                style={
                  isShowcase
                    ? {
                        strokeDasharray: lineLen,
                        strokeDashoffset: lineLen * (1 - layerProg),
                        transition: "stroke-dashoffset 0.6s ease-out",
                        opacity: 0.4 + layerProg * 0.6,
                      }
                    : { opacity: 0.65 }
                }
              />
            );
          })}

          {/* ═══ Input grid (28×28 circles) ═══ */}
          <g
            style={{ opacity: getNodeOpacity(-1), transition: "opacity 0.5s" }}
          >
            {inputPositions.map((pt, i) => {
              const row = Math.floor(i / 28);
              const col = i % 28;
              const value = inputGrid[row][col];
              return (
                <circle
                  key={`in-${i}`}
                  cx={pt.x}
                  cy={pt.y}
                  r={2.8}
                  fill={nodeColor(value)}
                  style={
                    value > 0.3
                      ? {
                          filter: `drop-shadow(0 0 ${2 + value * 4}px rgba(0,255,157,${value * 0.5}))`,
                        }
                      : undefined
                  }
                />
              );
            })}
          </g>

          {/* ═══ Hidden Layer 1 ═══ */}
          <g style={{ opacity: getNodeOpacity(0), transition: "opacity 0.5s" }}>
            {hidden1Positions.map((pt, i) => (
              <circle
                key={`h1-${i}`}
                cx={pt.x}
                cy={pt.y}
                r={3.2}
                fill={nodeColor(hidden1[i])}
                stroke={
                  hidden1[i] > 0.4
                    ? "rgba(0,255,157,0.3)"
                    : "rgba(255,255,255,0.06)"
                }
                strokeWidth={0.5}
                style={
                  hidden1[i] > 0.2
                    ? {
                        filter: `drop-shadow(0 0 ${3 + hidden1[i] * 6}px rgba(0,255,157,${hidden1[i] * 0.4}))`,
                      }
                    : undefined
                }
              />
            ))}
          </g>

          {/* ═══ Hidden Layer 2 ═══ */}
          <g style={{ opacity: getNodeOpacity(1), transition: "opacity 0.5s" }}>
            {hidden2Positions.map((pt, i) => (
              <circle
                key={`h2-${i}`}
                cx={pt.x}
                cy={pt.y}
                r={5}
                fill={nodeColor(hidden2[i])}
                stroke={
                  hidden2[i] > 0.4
                    ? "rgba(0,255,157,0.35)"
                    : "rgba(255,255,255,0.08)"
                }
                strokeWidth={0.6}
                style={
                  hidden2[i] > 0.2
                    ? {
                        filter: `drop-shadow(0 0 ${3 + hidden2[i] * 7}px rgba(0,255,157,${hidden2[i] * 0.45}))`,
                      }
                    : undefined
                }
              />
            ))}
          </g>

          {/* ═══ Output nodes — digit labels + probability ═══ */}
          <g style={{ opacity: getNodeOpacity(2), transition: "opacity 0.5s" }}>
            {outputPositions.map((pt, i) => {
              const prob = output[i];
              const isPredicted = i === predicted && prob > 0.1;
              const r = isPredicted ? 18 : 14;
              return (
                <g key={`out-${i}`}>
                  {isPredicted && (
                    <circle
                      cx={pt.x}
                      cy={pt.y}
                      r={r + 4}
                      fill="none"
                      stroke="rgba(0,255,157,0.4)"
                      strokeWidth={1.5}
                      filter="url(#nn-glow-strong)"
                    >
                      <animate
                        attributeName="r"
                        values={`${r + 3};${r + 7};${r + 3}`}
                        dur="2s"
                        repeatCount="indefinite"
                      />
                      <animate
                        attributeName="opacity"
                        values="0.6;0.2;0.6"
                        dur="2s"
                        repeatCount="indefinite"
                      />
                    </circle>
                  )}
                  <circle
                    cx={pt.x}
                    cy={pt.y}
                    r={r}
                    fill={nodeColor(prob)}
                    stroke={
                      isPredicted
                        ? "rgba(0,255,157,0.6)"
                        : "rgba(255,255,255,0.12)"
                    }
                    strokeWidth={isPredicted ? 1.5 : 0.8}
                    filter={prob > 0.2 ? "url(#nn-glow)" : undefined}
                  />
                  <text
                    x={pt.x}
                    y={pt.y + 1}
                    textAnchor="middle"
                    dominantBaseline="middle"
                    fill={prob > 0.5 ? "#0a0a0a" : "#e0e0e0"}
                    fontSize={isPredicted ? "14" : "11"}
                    fontWeight="700"
                    fontFamily="JetBrains Mono, monospace"
                  >
                    {i}
                  </text>
                  <text
                    x={pt.x + r + 8}
                    y={pt.y + 1}
                    textAnchor="start"
                    dominantBaseline="middle"
                    fill={isPredicted ? "#00ff9d" : "rgba(255,255,255,0.4)"}
                    fontSize={isPredicted ? "11" : "9"}
                    fontWeight={isPredicted ? "700" : "400"}
                    fontFamily="JetBrains Mono, monospace"
                  >
                    {(prob * 100).toFixed(1)}%
                  </text>
                </g>
              );
            })}
          </g>

          {/* ═══ Math equations between layers ═══ */}
          {EQUATIONS.map((eq, i) => (
            <g
              key={`eq-${i}`}
              opacity={
                isShowcase ? (getLayerProgress(i) > 0 ? 0.85 : 0.2) : 0.75
              }
              style={{ transition: "opacity 0.5s" }}
            >
              <rect
                x={eq.x - 90}
                y={VB_H - 38}
                width={180}
                height={22}
                rx={11}
                fill="rgba(0,0,0,0.6)"
                stroke="rgba(255,255,255,0.06)"
                strokeWidth={0.5}
              />
              <text
                x={eq.x}
                y={VB_H - 23}
                textAnchor="middle"
                dominantBaseline="middle"
                fill="#8ecae6"
                fontSize="10"
                fontFamily="'Cambria Math', 'Latin Modern Math', Georgia, serif"
                fontStyle="italic"
                letterSpacing="0.03em"
              >
                {eq.label}
              </text>
            </g>
          ))}

          {/* ═══ Layer labels ═══ */}
          {LAYER_LABELS.map((entry) => {
            const active =
              isShowcase && entry.idx >= 0 && getLayerProgress(entry.idx) > 0;
            const showActive = active || (!isShowcase && entry.idx >= 0);
            return (
              <g key={entry.label}>
                <text
                  x={entry.x}
                  y={30}
                  textAnchor="middle"
                  fill={
                    showActive
                      ? "rgba(255,255,255,0.8)"
                      : "rgba(255,255,255,0.35)"
                  }
                  fontSize="11"
                  fontWeight="600"
                  fontFamily="Inter, sans-serif"
                  letterSpacing="0.04em"
                  style={{ transition: "fill 0.5s ease" }}
                >
                  {entry.label}
                </text>
                <text
                  x={entry.x}
                  y={44}
                  textAnchor="middle"
                  fill={
                    showActive ? "rgba(0,255,157,0.6)" : "rgba(255,255,255,0.2)"
                  }
                  fontSize="9"
                  fontFamily="JetBrains Mono, monospace"
                  style={{ transition: "fill 0.5s ease" }}
                >
                  {entry.count}
                </text>
              </g>
            );
          })}

          {/* ═══ Weight count annotations ═══ */}
          <g opacity="0.35">
            <text
              x={(INPUT_X + H1_X) / 2}
              y={TOP_Y - 2}
              textAnchor="middle"
              fill="#ffd700"
              fontSize="8"
              fontFamily="JetBrains Mono, monospace"
            >
              {(784 * 96).toLocaleString()} weights
            </text>
            <text
              x={(H1_X + H2_X) / 2}
              y={TOP_Y - 2}
              textAnchor="middle"
              fill="#ffd700"
              fontSize="8"
              fontFamily="JetBrains Mono, monospace"
            >
              {(96 * 48).toLocaleString()} weights
            </text>
            <text
              x={(H2_X + OUTPUT_X) / 2}
              y={TOP_Y - 2}
              textAnchor="middle"
              fill="#ffd700"
              fontSize="8"
              fontFamily="JetBrains Mono, monospace"
            >
              {(48 * 10).toLocaleString()} weights
            </text>
          </g>
        </svg>
      </div>

      {/* Educational footer */}
      <div className="nn-viz-footer">
        <span className="nn-viz-footer-note">
          Each line represents a learned weight · Brightness = activation
          strength · Green = positive signal
        </span>
      </div>
    </div>
  );
}
