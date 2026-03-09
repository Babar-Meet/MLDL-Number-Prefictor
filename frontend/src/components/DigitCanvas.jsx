import { useEffect, useRef } from "react";

const GRID_SIZE = 28;
const CELL_SIZE = 12;
const CANVAS_SIZE = GRID_SIZE * CELL_SIZE;

function clamp(value, min, max) {
  return Math.max(min, Math.min(max, value));
}

function activationColor(value) {
  const intensity = clamp(value, 0, 1);
  const hue = 28 + intensity * 18;
  const saturation = 48 + intensity * 42;
  const lightness = 10 + intensity * 72;
  return `hsl(${hue} ${saturation}% ${lightness}%)`;
}

function cloneGrid(grid) {
  return grid.map((row) => [...row]);
}

function applyBrush(grid, fromPoint, toPoint, brushRadius, brushStrength) {
  const nextGrid = cloneGrid(grid);
  const stepCount =
    Math.max(
      Math.abs(toPoint.col - fromPoint.col),
      Math.abs(toPoint.row - fromPoint.row),
      1,
    ) * 2;

  for (let step = 0; step <= stepCount; step += 1) {
    const progress = step / stepCount;
    const row = Math.round(
      fromPoint.row + (toPoint.row - fromPoint.row) * progress,
    );
    const col = Math.round(
      fromPoint.col + (toPoint.col - fromPoint.col) * progress,
    );

    for (
      let rowOffset = -brushRadius;
      rowOffset <= brushRadius;
      rowOffset += 1
    ) {
      for (
        let colOffset = -brushRadius;
        colOffset <= brushRadius;
        colOffset += 1
      ) {
        const targetRow = row + rowOffset;
        const targetCol = col + colOffset;

        if (
          targetRow < 0 ||
          targetRow >= GRID_SIZE ||
          targetCol < 0 ||
          targetCol >= GRID_SIZE
        ) {
          continue;
        }

        const distance = Math.hypot(rowOffset, colOffset);
        const falloff = Math.max(0, 1 - distance / (brushRadius + 0.75));
        if (falloff <= 0) {
          continue;
        }

        const currentValue = nextGrid[targetRow][targetCol];
        const paintedValue = currentValue + brushStrength * falloff * 0.42;
        nextGrid[targetRow][targetCol] = clamp(paintedValue, 0, 1);
      }
    }
  }

  return nextGrid;
}

export default function DigitCanvas({
  grid,
  onChange,
  brushRadius,
  brushStrength,
}) {
  const canvasRef = useRef(null);
  const paintingRef = useRef(false);
  const lastPointRef = useRef(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    const context = canvas?.getContext("2d");
    if (!canvas || !context) {
      return;
    }

    context.clearRect(0, 0, CANVAS_SIZE, CANVAS_SIZE);
    context.fillStyle = "#120d1c";
    context.fillRect(0, 0, CANVAS_SIZE, CANVAS_SIZE);

    for (let row = 0; row < GRID_SIZE; row += 1) {
      for (let col = 0; col < GRID_SIZE; col += 1) {
        context.fillStyle = activationColor(grid[row][col]);
        context.fillRect(
          col * CELL_SIZE,
          row * CELL_SIZE,
          CELL_SIZE,
          CELL_SIZE,
        );
      }
    }

    context.strokeStyle = "rgba(255, 255, 255, 0.08)";
    context.lineWidth = 1;

    for (let index = 0; index <= GRID_SIZE; index += 1) {
      context.beginPath();
      context.moveTo(index * CELL_SIZE, 0);
      context.lineTo(index * CELL_SIZE, CANVAS_SIZE);
      context.stroke();

      context.beginPath();
      context.moveTo(0, index * CELL_SIZE);
      context.lineTo(CANVAS_SIZE, index * CELL_SIZE);
      context.stroke();
    }
  }, [grid]);

  useEffect(() => {
    const stopPainting = () => {
      paintingRef.current = false;
      lastPointRef.current = null;
    };

    window.addEventListener("pointerup", stopPainting);
    return () => window.removeEventListener("pointerup", stopPainting);
  }, []);

  function getPoint(event) {
    const bounds = canvasRef.current.getBoundingClientRect();
    const col = clamp(
      Math.floor(((event.clientX - bounds.left) / bounds.width) * GRID_SIZE),
      0,
      GRID_SIZE - 1,
    );
    const row = clamp(
      Math.floor(((event.clientY - bounds.top) / bounds.height) * GRID_SIZE),
      0,
      GRID_SIZE - 1,
    );
    return { row, col };
  }

  function paint(event) {
    if (!paintingRef.current) {
      return;
    }

    const point = getPoint(event);
    const startPoint = lastPointRef.current ?? point;
    const nextGrid = applyBrush(
      grid,
      startPoint,
      point,
      brushRadius,
      brushStrength,
    );
    lastPointRef.current = point;
    onChange(nextGrid);
  }

  return (
    <div className="glass-panel flex flex-col gap-4 p-5">
      <div className="flex items-center justify-between gap-3">
        <div className="flex items-center gap-2">
          <span className="section-label">
            <span className="inline-block h-2 w-2 rounded-full bg-[#06d6a0] shadow-[0_0_6px_rgba(6,214,160,0.5)]" />
            Input
          </span>
        </div>
        <span className="rounded-lg border border-white/6 bg-white/3 px-2.5 py-1 text-[0.65rem] font-medium text-white/45 mono">
          28 × 28
        </span>
      </div>

      <canvas
        ref={canvasRef}
        width={CANVAS_SIZE}
        height={CANVAS_SIZE}
        className="canvas-frame aspect-square w-full touch-none rounded-2xl"
        onPointerDown={(event) => {
          paintingRef.current = true;
          const point = getPoint(event);
          lastPointRef.current = point;
          onChange(applyBrush(grid, point, point, brushRadius, brushStrength));
        }}
        onPointerMove={paint}
        onPointerLeave={() => {
          paintingRef.current = false;
          lastPointRef.current = null;
        }}
      />

      <p className="text-[0.7rem] leading-relaxed text-white/35">
        Draw naturally — the canvas interpolates between pointer positions for
        smooth strokes.
      </p>
    </div>
  );
}
