import { Canvas } from "@react-three/fiber";
import { OrbitControls, Line, Text } from "@react-three/drei";

function clamp(v, lo, hi) {
  return Math.max(lo, Math.min(hi, v));
}

function glowColor(value) {
  const t = clamp(value ?? 0, 0, 1);
  const r = Math.round(51 + t * (0 - 51));
  const g = Math.round(51 + t * (255 - 51));
  const b = Math.round(51 + t * (157 - 51));
  return `rgb(${r},${g},${b})`;
}

/* Show subset of input (every 2nd pixel) to reduce clutter */
function buildInputNodes(grid) {
  const nodes = [];
  for (let row = 0; row < 28; row += 2) {
    for (let col = 0; col < 28; col += 2) {
      nodes.push({
        id: `${row}-${col}`,
        position: [-7, (6.5 - row * 0.5) * 0.7, (col * 0.5 - 6.5) * 0.7],
        activation: grid[row][col],
      });
    }
  }
  return nodes;
}

function buildLayerNodes(values, xPos, spacing) {
  return values.map((value, i) => ({
    id: `${xPos}-${i}`,
    position: [xPos, ((values.length - 1) * spacing) / 2 - i * spacing, 0],
    activation: value,
  }));
}

function NodeCloud({ nodes, radius }) {
  return nodes.map((node) => (
    <mesh key={node.id} position={node.position}>
      <sphereGeometry
        args={[radius + node.activation * radius * 0.8, 12, 12]}
      />
      <meshStandardMaterial
        color={glowColor(node.activation)}
        emissive={glowColor(node.activation)}
        emissiveIntensity={0.7 + node.activation * 2}
        toneMapped={false}
      />
    </mesh>
  ));
}

function LayerLabel({ position, text, sub }) {
  return (
    <group position={position}>
      <Text
        fontSize={0.35}
        color="#ffffff"
        anchorX="center"
        anchorY="bottom"
        font={undefined}
      >
        {text}
      </Text>
      {sub && (
        <Text
          position={[0, -0.45, 0]}
          fontSize={0.22}
          color="#00ff9d"
          anchorX="center"
          anchorY="bottom"
          font={undefined}
        >
          {sub}
        </Text>
      )}
    </group>
  );
}

/* Build representative dense connections (sampled) */
function buildEdges(srcNodes, tgtNodes, maxEdges) {
  const edges = [];
  const step = Math.max(
    1,
    Math.floor(srcNodes.length / Math.ceil(maxEdges / tgtNodes.length)),
  );
  for (let t = 0; t < tgtNodes.length; t++) {
    for (let s = 0; s < srcNodes.length; s += step) {
      edges.push({
        source: srcNodes[s].position,
        target: tgtNodes[t].position,
        signal: srcNodes[s].activation * tgtNodes[t].activation,
      });
    }
  }
  return edges;
}

export default function NetworkScene3D({ analysis, edgeRevealProgress }) {
  const inputGrid =
    analysis?.layers?.[0]?.grid ??
    Array.from({ length: 28 }, () => Array(28).fill(0));
  const hidden1 = analysis?.layers?.[1]?.activations ?? Array(96).fill(0);
  const hidden2 = analysis?.layers?.[2]?.activations ?? Array(48).fill(0);
  const output = analysis?.prediction?.probabilities ?? Array(10).fill(0);

  const inputNodes = buildInputNodes(inputGrid);
  const hidden1Nodes = buildLayerNodes(hidden1, -2.5, 0.09);
  const hidden2Nodes = buildLayerNodes(hidden2, 1.5, 0.16);
  const outputNodes = buildLayerNodes(output, 5, 0.6);

  const isShowcase = edgeRevealProgress != null;

  // Dense edge webs between layers (sampled)
  const edgesL0 = buildEdges(inputNodes, hidden1Nodes, 120);
  const edgesL1 = buildEdges(hidden1Nodes, hidden2Nodes, 100);
  const edgesL2 = buildEdges(hidden2Nodes, outputNodes, 60);

  const allDenseEdges = [
    ...edgesL0.map((e) => ({ ...e, layer: 0 })),
    ...edgesL1.map((e) => ({ ...e, layer: 1 })),
    ...edgesL2.map((e) => ({ ...e, layer: 2 })),
  ];

  const visibleEdges = isShowcase
    ? allDenseEdges.filter(
        (_, i) => i < Math.ceil(edgeRevealProgress * allDenseEdges.length),
      )
    : allDenseEdges;

  return (
    <div className="nn-viz-container">
      <div className="nn-viz-header">
        <div className="nn-viz-title">
          <span className="nn-viz-dot" />
          3D Network View
        </div>
        <span className="nn-arch-tag">
          {isShowcase ? "auto-rotating" : "drag to orbit"}
        </span>
      </div>

      <div className="nn-3d-canvas">
        <Canvas camera={{ position: [10, 3, 10], fov: 40 }}>
          <color attach="background" args={["#0d1117"]} />
          <fog attach="fog" args={["#0d1117", 30, 50]} />
          <ambientLight intensity={1.2} />
          <pointLight position={[8, 10, 8]} intensity={80} color="#00ff9d" />
          <pointLight position={[-8, -6, 8]} intensity={50} color="#7c3aed" />
          <pointLight position={[0, 6, -8]} intensity={40} color="#3b82f6" />
          <pointLight position={[0, -8, 6]} intensity={30} color="#ffffff" />

          {/* Input layer — reduced subset */}
          <NodeCloud nodes={inputNodes} radius={0.06} />
          {/* Hidden 1 */}
          <NodeCloud nodes={hidden1Nodes} radius={0.06} />
          {/* Hidden 2 */}
          <NodeCloud nodes={hidden2Nodes} radius={0.1} />
          {/* Output */}
          <NodeCloud nodes={outputNodes} radius={0.18} />

          {/* Layer labels */}
          <LayerLabel position={[-7, 5.5, 0]} text="Input" sub="784" />
          <LayerLabel position={[-2.5, 5.5, 0]} text="Hidden 1" sub="96" />
          <LayerLabel position={[1.5, 5.5, 0]} text="Hidden 2" sub="48" />
          <LayerLabel position={[5, 5.5, 0]} text="Output" sub="10" />

          {/* Edges */}
          {visibleEdges.map((edge, i) => {
            const mag = clamp(Math.abs(edge.signal) * 5, 0, 1);
            return (
              <Line
                key={i}
                points={[edge.source, edge.target]}
                color={edge.signal >= 0 ? "#00ff9d" : "#ef4444"}
                lineWidth={0.5 + mag * 2}
                opacity={0.15 + mag * 0.5}
                transparent
              />
            );
          })}

          <OrbitControls
            enablePan={false}
            minDistance={8}
            maxDistance={24}
            autoRotate
            autoRotateSpeed={isShowcase ? 1.5 : 0.5}
          />
        </Canvas>
      </div>
    </div>
  );
}
