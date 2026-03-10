export const GRID_SIZE = 28;

function clamp(value, min, max) {
  return Math.max(min, Math.min(max, value));
}

function cloneGrid(grid) {
  return grid.map((row) => [...row]);
}

function createEmptyGrid() {
  return Array.from({ length: GRID_SIZE }, () => Array(GRID_SIZE).fill(0));
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

function drawPath(grid, path, brushRadius, brushStrength) {
  let g = grid;
  for (let i = 0; i < path.length - 1; i++) {
    g = applyBrush(g, path[i], path[i + 1], brushRadius, brushStrength);
  }
  return g;
}

export function buildHandwrittenDigit(digit) {
  let grid = createEmptyGrid();
  // Ensure brush is natural looking
  const radius = Math.random() > 0.5 ? 2 : 1;
  const strength = Math.random() * 0.6 + 0.8;

  const padScale = (val, offset) => val + offset + (Math.random() * 2 - 1);

  const generatePath = (points) => {
    return points.map(([r, c]) => ({
      row: padScale(r, 0),
      col: padScale(c, 0),
    }));
  };

  let paths = [];
  // Using simplified coordinate points for typical MNIST styles [row, col]
  switch (digit) {
    case 0:
      paths.push(
        generatePath([
          [6, 14],
          [8, 9],
          [14, 7],
          [20, 10],
          [21, 15],
          [19, 20],
          [13, 21],
          [7, 18],
          [6, 14],
        ]),
      );
      break;
    case 1:
      paths.push(
        generatePath([
          [8, 12],
          [5, 14],
          [22, 14],
        ]),
      );
      break;
    case 2:
      paths.push(
        generatePath([
          [8, 8],
          [5, 14],
          [8, 20],
          [14, 16],
          [22, 8],
          [22, 20],
        ]),
      );
      break;
    case 3:
      paths.push(
        generatePath([
          [7, 8],
          [5, 14],
          [9, 20],
          [13, 16],
          [13, 13],
        ]),
      );
      paths.push(
        generatePath([
          [13, 13],
          [14, 19],
          [20, 20],
          [23, 14],
          [20, 8],
        ]),
      );
      break;
    case 4:
      paths.push(
        generatePath([
          [5, 18],
          [15, 7],
          [15, 22],
        ]),
      );
      paths.push(
        generatePath([
          [9, 17],
          [22, 17],
        ]),
      );
      break;
    case 5:
      paths.push(
        generatePath([
          [6, 20],
          [7, 10],
          [13, 10],
          [11, 16],
          [16, 21],
          [21, 18],
          [22, 10],
        ]),
      );
      break;
    case 6:
      paths.push(
        generatePath([
          [5, 18],
          [14, 9],
          [21, 10],
          [23, 15],
          [20, 20],
          [15, 19],
          [13, 13],
          [14, 9],
        ]),
      );
      break;
    case 7:
      paths.push(
        generatePath([
          [6, 7],
          [6, 21],
          [13, 16],
          [22, 12],
        ]),
      );
      break;
    case 8:
      paths.push(
        generatePath([
          [13, 14],
          [9, 9],
          [5, 14],
          [9, 19],
          [13, 14],
          [19, 9],
          [23, 14],
          [19, 19],
          [13, 14],
        ]),
      );
      break;
    case 9:
      paths.push(
        generatePath([
          [15, 14],
          [13, 9],
          [8, 9],
          [5, 14],
          [9, 19],
          [15, 19],
          [22, 14],
        ]),
      );
      break;
    default:
      paths.push(
        generatePath([
          [7, 14],
          [14, 14],
          [21, 14],
        ]),
      );
      break;
  }

  // Slight rotation & random scale for variation
  const angle = (Math.random() - 0.5) * 0.4;
  const sinA = Math.sin(angle);
  const cosA = Math.cos(angle);
  const scale = 0.8 + Math.random() * 0.3;

  for (let path of paths) {
    for (let point of path) {
      const dr = (point.row - 14) * scale;
      const dc = (point.col - 14) * scale;
      point.row = 14 + dr * cosA - dc * sinA;
      point.col = 14 + dr * sinA + dc * cosA;
    }
    // Interpolate points for smoother curve
    const smoothPath = [];
    if (path.length > 2) {
      for (let i = 0; i < path.length - 1; i++) {
        smoothPath.push(path[i]);
        smoothPath.push({
          row: (path[i].row + path[i + 1].row) / 2,
          col: (path[i].col + path[i + 1].col) / 2,
        });
      }
      smoothPath.push(path[path.length - 1]);
    } else {
      smoothPath.push(...path);
    }
    grid = drawPath(grid, smoothPath, radius, strength);
  }

  // Blur slightly
  let blurred = createEmptyGrid();
  for (let r = 0; r < GRID_SIZE; r++) {
    for (let c = 0; c < GRID_SIZE; c++) {
      let sum = 0;
      let w = 0;
      for (let dr = -1; dr <= 1; dr++) {
        for (let dc = -1; dc <= 1; dc++) {
          if (
            r + dr >= 0 &&
            r + dr < GRID_SIZE &&
            c + dc >= 0 &&
            c + dc < GRID_SIZE
          ) {
            const weight =
              dr === 0 && dc === 0
                ? 0.5
                : Math.abs(dr) + Math.abs(dc) === 1
                  ? 0.1
                  : 0.025;
            sum += grid[r + dr][c + dc] * weight;
            w += weight;
          }
        }
      }
      blurred[r][c] = claimValue(sum / w);
    }
  }

  return blurred;
}

function claimValue(val) {
  return val > 1 ? 1 : val;
}
