import { scaleLinear } from "d3-scale";
import { motion } from "framer-motion";

function clamp(value, min, max) {
  return Math.max(min, Math.min(max, value));
}

function heatColor(value, maxMagnitude) {
  const normalised =
    maxMagnitude === 0 ? 0 : clamp(Math.abs(value) / maxMagnitude, 0, 1);
  if (value >= 0) {
    return `hsl(160 ${55 + normalised * 25}% ${14 + normalised * 50}%)`;
  }
  return `hsl(350 ${65 + normalised * 15}% ${14 + normalised * 50}%)`;
}

const stagger = {
  hidden: { opacity: 0 },
  visible: {
    opacity: 1,
    transition: { staggerChildren: 0.04, delayChildren: 0.05 },
  },
};
const fadeUp = {
  hidden: { opacity: 0, y: 12 },
  visible: {
    opacity: 1,
    y: 0,
    transition: { duration: 0.4, ease: [0.16, 1, 0.3, 1] },
  },
};

export default function ActivationCharts({ model, analysis }) {
  const probabilities =
    analysis?.prediction?.probabilities ?? Array(10).fill(0);
  const topContributors = analysis?.topContributors ?? [];
  const layerStats = analysis?.layers ?? [];
  const outputMatrix = model?.weights?.[2]?.matrix ?? [];
  const outputMaxMagnitude = outputMatrix.length
    ? Math.max(...outputMatrix.flat().map((value) => Math.abs(value)))
    : 1;

  const probabilityScale = scaleLinear().domain([0, 1]).range([0, 100]);

  return (
    <div className="grid gap-6 xl:grid-cols-[1.1fr_0.9fr]">
      {/* Confidence distribution */}
      <div className="glass-panel p-5">
        <div className="mb-5 flex items-center justify-between gap-3">
          <span className="section-label">
            <span className="inline-block h-2 w-2 rounded-full bg-[#f59e0b] shadow-[0_0_6px_rgba(245,158,11,0.5)]" />
            Confidence
          </span>
          <span className="rounded-lg border border-white/6 bg-white/3 px-2.5 py-1 text-[0.65rem] font-medium text-white/40 mono">
            digit {analysis?.prediction?.digit ?? "–"}
          </span>
        </div>

        <motion.div
          className="space-y-2.5"
          variants={stagger}
          initial="hidden"
          animate={analysis ? "visible" : "hidden"}
        >
          {probabilities.map((value, index) => (
            <motion.div key={index} variants={fadeUp} className="space-y-1">
              <div className="flex items-center justify-between text-xs">
                <span className="mono font-medium text-white/55">{index}</span>
                <span className="mono text-white/45">
                  {(value * 100).toFixed(1)}%
                </span>
              </div>
              <div className="confidence-bar">
                <div
                  className="confidence-fill"
                  style={{ width: `${probabilityScale(value)}%` }}
                />
              </div>
            </motion.div>
          ))}
        </motion.div>

        {/* Layer stats */}
        <div className="mt-5 grid gap-2.5 sm:grid-cols-2 xl:grid-cols-4">
          {layerStats.map((layer) => (
            <div key={layer.name} className="inner-card p-3">
              <p className="text-[0.6rem] font-semibold uppercase tracking-[0.1em] text-white/30 mb-2">
                {layer.name}
              </p>
              <p className="text-sm font-bold text-white mono">
                {layer.size}{" "}
                <span className="text-white/35 font-normal">neurons</span>
              </p>
              <div className="mt-2.5 space-y-1 text-[0.7rem] text-white/40">
                <div className="flex justify-between">
                  <span>mean</span>
                  <span className="mono text-white/55">
                    {layer.stats.mean.toFixed(3)}
                  </span>
                </div>
                <div className="flex justify-between">
                  <span>max</span>
                  <span className="mono text-white/55">
                    {layer.stats.max.toFixed(3)}
                  </span>
                </div>
                <div className="flex justify-between">
                  <span>sparsity</span>
                  <span className="mono text-white/55">
                    {(layer.sparsity * 100).toFixed(1)}%
                  </span>
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Weight matrix + contributors */}
      <div className="glass-panel p-5">
        <div className="mb-5 flex items-center justify-between gap-3">
          <span className="section-label">
            <span className="inline-block h-2 w-2 rounded-full bg-[#7c3aed] shadow-[0_0_6px_rgba(124,58,237,0.5)]" />
            Weight Matrix
          </span>
          <span className="rounded-lg border border-white/6 bg-white/3 px-2.5 py-1 text-[0.65rem] font-medium text-white/40 mono">
            H₂ → Out
          </span>
        </div>

        <div className="overflow-x-auto rounded-xl border border-white/5 bg-[#060a12] p-2.5">
          <svg
            viewBox={`0 0 ${outputMatrix[0]?.length * 11 + 48 || 64} ${outputMatrix.length * 16 + 32 || 64}`}
            className="min-w-[520px]"
          >
            {outputMatrix.map((row, rowIndex) =>
              row.map((value, columnIndex) => (
                <rect
                  key={`${rowIndex}-${columnIndex}`}
                  x={40 + columnIndex * 11}
                  y={12 + rowIndex * 16}
                  width="10"
                  height="14"
                  rx="2"
                  fill={heatColor(value, outputMaxMagnitude)}
                />
              )),
            )}
            {outputMatrix.map((_, rowIndex) => (
              <text
                key={`label-${rowIndex}`}
                x="8"
                y={24 + rowIndex * 16}
                fill="rgba(255,255,255,0.35)"
                fontSize="9"
                fontFamily="JetBrains Mono, monospace"
              >
                {rowIndex}
              </text>
            ))}
          </svg>
        </div>

        {/* Top contributors */}
        <div className="mt-5 space-y-2">
          <p className="text-[0.6rem] font-semibold uppercase tracking-[0.1em] text-white/30 mb-2">
            Top contributors
          </p>
          {topContributors.map((contributor) => (
            <div key={contributor.index} className="inner-card p-3 text-xs">
              <div className="flex items-center justify-between gap-3">
                <span className="font-semibold text-white/80 mono">
                  H₂[{contributor.index}]
                </span>
                <span className="mono text-[#06d6a0]">
                  {contributor.contribution.toFixed(4)}
                </span>
              </div>
              <div className="mt-1.5 flex items-center justify-between gap-3 text-white/35">
                <span>act {contributor.activation.toFixed(4)}</span>
                <span>wt {contributor.weight.toFixed(4)}</span>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
