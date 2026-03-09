import { motion, AnimatePresence } from "framer-motion";

export default function PredictionHistory({ history }) {
  return (
    <div className="glass-panel p-5">
      <div className="mb-5 flex items-center justify-between gap-3">
        <span className="section-label">
          <span className="inline-block h-2 w-2 rounded-full bg-[#ec4899] shadow-[0_0_6px_rgba(236,72,153,0.5)]" />
          History
        </span>
        <span className="rounded-lg border border-white/6 bg-white/3 px-2.5 py-1 text-[0.65rem] font-medium text-white/40 mono">
          {history.length} entries
        </span>
      </div>

      <div className="space-y-2.5">
        {history.length === 0 && (
          <div className="rounded-xl border border-dashed border-white/8 bg-white/[0.015] p-5 text-center text-xs text-white/35">
            No predictions yet — analyze a drawing to start the live feed.
          </div>
        )}

        <AnimatePresence initial={false}>
          {history.map((entry) => (
            <motion.div
              key={entry.id}
              initial={{ opacity: 0, y: 12, scale: 0.97 }}
              animate={{ opacity: 1, y: 0, scale: 1 }}
              exit={{ opacity: 0, scale: 0.95 }}
              transition={{ duration: 0.35, ease: [0.16, 1, 0.3, 1] }}
              className="inner-card p-4"
            >
              <div className="flex items-center justify-between gap-3">
                <div>
                  <p className="text-[0.65rem] text-white/30">
                    {new Date(entry.createdAt).toLocaleTimeString()}
                  </p>
                  <p className="text-base font-bold text-white mono">
                    {entry.digit}
                  </p>
                </div>
                <div className="text-right">
                  <p className="text-[0.6rem] uppercase tracking-[0.1em] text-white/25">
                    confidence
                  </p>
                  <p className="text-base font-bold text-white mono">
                    {(entry.confidence * 100).toFixed(1)}%
                  </p>
                </div>
              </div>

              <div className="mt-3 grid grid-cols-10 gap-1">
                {entry.probabilities.map((value, index) => (
                  <div
                    key={index}
                    className="flex flex-col items-center gap-0.5"
                  >
                    <div className="flex h-8 w-full items-end rounded bg-white/4">
                      <div
                        className="w-full rounded confidence-fill"
                        style={{
                          height: `${Math.max(8, value * 100)}%`,
                          animationDelay: `${index * 0.1}s`,
                        }}
                      />
                    </div>
                    <span className="text-[9px] text-white/30 mono">
                      {index}
                    </span>
                  </div>
                ))}
              </div>
            </motion.div>
          ))}
        </AnimatePresence>
      </div>
    </div>
  );
}
