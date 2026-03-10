# Ultimate Neural Network Visualizer

Interactive MNIST digit classifier with a React visualization dashboard, an Express plus Socket.IO bridge, a FastAPI plus PyTorch inference server, and optional MongoDB Atlas persistence for prediction history. The app lets you draw a 28x28 digit, inspect live activations and dominant connections, adjust layer scales and output bias, and optionally run a single labeled backpropagation step for demo retraining.

## Stack

- Frontend: React 19, Vite, Tailwind CSS v4, D3 scale utilities, React Three Fiber
- Backend bridge: Node.js, Express 5, Socket.IO, MongoDB Atlas
- ML service: FastAPI, PyTorch, torchvision
- Model: 784 -> 96 -> 48 -> 10 feedforward MNIST classifier

## What is implemented

- Smooth 28x28 drawing canvas with preset demo digits
- Realtime network flow view with neuron activations and highlighted active edges
- Confidence bars, layer statistics, output weight heatmap, and prediction history
- MongoDB-backed prediction history with automatic in-memory fallback if Atlas is unavailable
- Optional 3D network scene with orbit controls
- Manual layer-scale and output-bias controls to inspect prediction sensitivity
- Backpropagation metrics and strongest gradient-driven weight update candidates
- One-step demo retraining endpoint for labeled user inputs
- Production frontend build with the Three.js view code-split into a separate chunk

## Project layout

- Frontend app: [frontend/src/App.jsx](/b:/_Git/DEV/Babar-Meet/MLDL%20Number%20Prefictor/frontend/src/App.jsx)
- Python model utilities: [backend/model_utils.py](/b:/_Git/DEV/Babar-Meet/MLDL%20Number%20Prefictor/backend/model_utils.py)
- FastAPI ML server: [backend/ml_server.py](/b:/_Git/DEV/Babar-Meet/MLDL%20Number%20Prefictor/backend/ml_server.py)
- Express bridge: [backend/server.js](/b:/_Git/DEV/Babar-Meet/MLDL%20Number%20Prefictor/backend/server.js)
- Training entrypoint: [train_model.py](/b:/_Git/DEV/Babar-Meet/MLDL%20Number%20Prefictor/train_model.py)

## Local setup

### Windows quick start

For a faculty demo on Windows, use the batch files in the project root:

- `setup.bat`: creates `.env`, installs Python and Node dependencies, prepares the model if needed, and builds the frontend
- `launch_showcase.bat`: starts the FastAPI service and the Express showcase app, then opens the browser

Recommended flow:

```bat
setup.bat
launch_showcase.bat
```

`setup.bat` also offers to launch the project immediately after setup finishes.

### 0. Environment variables

Create a `.env` file from `.env.example` and set the Atlas password in `MONGODB_URI`.

Important values for this project:

- Database name: `NumberPredictor`
- Username: `babarmeet`
- Atlas host: `numberpredictor.wu2qung.mongodb.net`

The backend intentionally falls back to in-memory history if `MONGODB_URI` is missing or Atlas is temporarily unreachable, so the demo still runs.

### 1. Python dependencies

Use the project interpreter or install into your preferred environment:

```powershell
python -m pip install -r backend/requirements.txt
```

### 2. Node dependencies

```powershell
cd backend
npm install
cd ../frontend
npm install
```

### 3. Train the model once

This generates:

- `backend/mnist_visualizer_model.pth`
- `backend/model_snapshot.json`

```powershell
python train_model.py
```

## Run locally

Open three terminals from the project root.

### Terminal 1: FastAPI ML server

```powershell
python -m uvicorn backend.ml_server:app --host 127.0.0.1 --port 8000
```

### Terminal 2: Express plus Socket.IO bridge

```powershell
cd backend
node server.js
```

If Atlas is configured correctly, `GET /api/health` will report the database mode as `mongodb-atlas`.

### Terminal 3: Frontend dev server

```powershell
cd frontend
npm run dev
```

Then open the Vite URL, normally `http://localhost:5173`.

## Production build

Frontend:

```powershell
cd frontend
npm run build
```

The built frontend is emitted to `frontend/dist` and the Express server is already configured to serve that directory when it exists.

## Docker

This repo now includes a two-container setup optimized for a smaller build context and better disk use:

- `backend/Dockerfile.web`: multi-stage build that compiles the frontend once and serves it from the Node bridge
- `backend/Dockerfile.ml`: slim Python image for FastAPI plus PyTorch inference
- `.dockerignore`: excludes raw MNIST data, local environments, and node_modules from the Docker context

### Prepare the environment

```powershell
copy .env.example .env
```

Then edit `.env` and replace `<db_password>` in `MONGODB_URI` with your Atlas password.

### Run with Docker Compose

```powershell
docker compose up --build
```

Services:

- Showcase app: `http://localhost:4000`
- FastAPI ML service: `http://localhost:8000`

Notes:

- Atlas stays outside Docker, which keeps the local stack much lighter than running a full MongoDB container.
- The raw MNIST dataset is not copied into the image, so image builds stay smaller and faster.
- The trained `backend/mnist_visualizer_model.pth` file is copied into the ML image so the service does not need to retrain on container startup.

## API summary

Node bridge:

- `GET /api/health`
- `GET /api/model`
- `GET /api/history`
- `POST /api/analyze`
- `POST /api/train-step`

FastAPI service:

- `GET /health`
- `GET /model`
- `POST /analyze`
- `POST /train-step`

Socket events:

- `analyze-digit`
- `train-digit`
- `prediction-history`
- `analysis-complete`
- `model-refresh-required`

## Deployment notes

### Frontend on Vercel

- Root directory: `frontend`
- Build command: `npm run build`
- Output directory: `dist`
- Set `VITE_API_BASE_URL` and `VITE_SOCKET_URL` to your deployed backend URL

### Backend on Render or similar

You can deploy the Node and Python services separately, or containerize them together.

Recommended simple split:

1. Deploy FastAPI service from the repo root with start command:

```bash
python -m uvicorn backend.ml_server:app --host 0.0.0.0 --port $PORT
```

2. Deploy Node bridge from `backend` with start command:

```bash
node server.js
```

3. Set `PYTHON_SERVICE_URL` on the Node service to the FastAPI deployment URL.
4. Set `MONGODB_URI`, `MONGODB_DB_NAME`, and `MONGODB_COLLECTION` on the Node service for Atlas-backed history.

### Single-stack Docker deployment

If your host supports multi-container deployments, use the included compose file as the base layout:

- One Node container serving the built frontend and Socket.IO bridge
- One Python container serving FastAPI inference
- One managed MongoDB Atlas cluster for persistent history

## Validation performed

- Frontend production build completed successfully
- Frontend lint completed successfully
- Model training completed successfully with test accuracy of 96.72%
- `GET /api/health` returned `ok`
- `POST /api/analyze` returned the full structured visualization payload through the Node bridge
- Atlas integration is wired into the backend with fallback behavior if the database is unavailable

## Known tradeoff

The optional 3D scene depends on Three.js and still ships a large vendor chunk. It is lazy-loaded and isolated from the main app bundle, which keeps the initial dashboard load much smaller even though the 3D feature remains heavyweight.

---

## Project Explanation

This section provides a comprehensive explanation of the project architecture, how each component works, and all the parameters used throughout the system.

### 1. Folder Structure

```
MLDL Number Prefictor/
├── backend/                          # Server-side components
│   ├── model_utils.py               # PyTorch ML model definition, training & inference
│   ├── ml_server.py                  # FastAPI REST API server for ML inference
│   ├── server.js                     # Express + Socket.IO bridge server
│   ├── db.js                         # MongoDB Atlas database integration
│   ├── requirements.txt              # Python dependencies (PyTorch, FastAPI, etc.)
│   ├── package.json                  # Node.js dependencies
│   ├── Dockerfile.ml                  # Docker configuration for ML service
│   ├── Dockerfile.web                 # Docker configuration for web service
│   ├── mnist_visualizer_model.pth    # Trained PyTorch model weights
│   └── model_snapshot.json           # JSON export of model weights for frontend
├── frontend/                          # React visualization dashboard
│   ├── src/
│   │   ├── App.jsx                  # Main React application
│   │   ├── App.css                  # Styling with Tailwind CSS
│   │   ├── index.css                # Global styles
│   │   ├── main.jsx                 # React entry point
│   │   ├── components/
│   │   │   ├── DigitCanvas.jsx      # 28x28 drawing canvas for digit input
│   │   │   ├── NetworkGraph.jsx     # 2D neural network visualization (SVG)
│   │   │   ├── NetworkScene3D.jsx  # 3D network visualization (Three.js)
│   │   │   ├── NetworkPipeline3D.jsx # Step-by-step 3D pipeline animation
│   │   │   ├── ActivationCharts.jsx # Confidence bars & weight heatmap
│   │   │   └── PredictionHistory.jsx # Past predictions display
│   │   └── utils/
│   │       └── digitGenerator.js    # Generate demo handwritten digits
│   ├── package.json                  # Frontend dependencies
│   ├── vite.config.js                # Vite build configuration
│   └── index.html                    # HTML template
├── data/MNIST/                       # MNIST dataset (downloaded on first run)
├── docker-compose.yml                # Container orchestration
├── train_model.py                    # Script to train the neural network
├── improve_model.py                  # Advanced training with augmentation
├── setup.bat                         # Windows setup script
├── launch_showcase.bat               # Launch all services
└── .env                              # Environment variables
```

### 2. Technologies & Libraries Used

#### Backend (Python)

| File                     | Technology       | Purpose                                                  |
| ------------------------ | ---------------- | -------------------------------------------------------- |
| `backend/model_utils.py` | **PyTorch 2.8+** | Neural network model definition, training, and inference |
| `backend/model_utils.py` | **torchvision**  | MNIST dataset loading and transforms                     |
| `backend/model_utils.py` | **NumPy**        | Numerical computations                                   |
| `backend/ml_server.py`   | **FastAPI**      | REST API server with automatic docs                      |
| `backend/ml_server.py`   | **Pydantic**     | Request/response data validation                         |
| `backend/server.js`      | **Express 5**    | HTTP web server                                          |
| `backend/server.js`      | **Socket.IO**    | Real-time bidirectional communication                    |
| `backend/server.js`      | **Axios**        | HTTP client to call Python API                           |
| `backend/db.js`          | **MongoDB**      | Persistent prediction history storage                    |

#### Frontend (JavaScript/React)

| File                    | Technology            | Purpose                           |
| ----------------------- | --------------------- | --------------------------------- |
| `frontend/package.json` | **React 19**          | UI framework                      |
| `frontend/package.json` | **Vite 7**            | Build tool and development server |
| `frontend/package.json` | **Tailwind CSS 4**    | Utility-first CSS framework       |
| `frontend/package.json` | **React Three Fiber** | React renderer for Three.js       |
| `frontend/package.json` | **Three.js**          | 3D graphics library               |
| `frontend/package.json` | **Socket.IO Client**  | Real-time client for Socket.IO    |
| `frontend/package.json` | **Axios**             | HTTP client for API calls         |
| `frontend/package.json` | **D3 Scale**          | Data scaling for charts           |
| `frontend/package.json` | **Framer Motion**     | Animation library                 |
| `frontend/package.json` | **Lucide React**      | Icon library                      |

### 3. Neural Network Architecture

The model is a feedforward neural network trained on the MNIST dataset:

```
Input Layer (784) → Hidden Layer 1 (96) → Hidden Layer 2 (48) → Output Layer (10)
```

#### Layer Details

| Layer             | Neurons  | Weights    | Biases  | Activation            |
| ----------------- | -------- | ---------- | ------- | --------------------- |
| Input → Hidden₁   | 784 → 96 | 75,264     | 96      | ReLU                  |
| Hidden₁ → Hidden₂ | 96 → 48  | 4,608      | 48      | ReLU                  |
| Hidden₂ → Output  | 48 → 10  | 480        | 10      | Softmax               |
| **Total**         |          | **80,352** | **154** | **80,506 parameters** |

### 4. How Socket.IO Works

Socket.IO enables **real-time bidirectional communication** between the frontend and backend. Here's how it's used in this project:

#### Connection Flow

1. **Frontend connects** to Socket.IO server on page load
2. **Server maintains** persistent connection with each client
3. **Events flow** both directions without HTTP polling

#### Socket Events Used

| Event Name               | Direction       | Payload                                  | Description                              |
| ------------------------ | --------------- | ---------------------------------------- | ---------------------------------------- |
| `analyze-digit`          | Client → Server | `{pixels, adjustments, includeBackprop}` | Send drawn digit for analysis            |
| `train-digit`            | Client → Server | `{pixels, adjustments, label}`           | Send digit with label for training       |
| `prediction-history`     | Server → Client | `Array`                                  | Broadcast updated history to all clients |
| `analysis-complete`      | Server → Client | `{analysis}`                             | Send analysis result back                |
| `model-refresh-required` | Server → Client | -                                        | Notify clients model was updated         |

#### Why Socket.IO?

- **Instant updates**: No need to refresh or poll
- **Broadcast support**: All connected clients see new predictions
- **Acknowledgement support**: Can confirm message delivery
- **Fallback**: Works with HTTP polling if WebSocket fails

### 5. Key Functions and Parameters

#### Python Backend

##### `train_model()` in `backend/model_utils.py`

Trains the neural network on MNIST data:

```python
def train_model(num_epochs: int = 6, batch_size: int = 128, learning_rate: float = 0.001)
```

| Parameter       | Default | Range       | Description                                     |
| --------------- | ------- | ----------- | ----------------------------------------------- |
| `num_epochs`    | 6       | 1-20        | Number of complete passes through training data |
| `batch_size`    | 128     | 16-512      | Samples processed before weight update          |
| `learning_rate` | 0.001   | 0.0001-0.01 | Adam optimizer step size                        |

##### `analyze_digit()` in `backend/model_utils.py`

Performs inference with optional gradient computation:

```python
def analyze_digit(
    pixels: list[list[float]],      # 28x28 grid (values 0.0-1.0)
    adjustments: dict | None = None, # Layer scaling & bias overrides
    include_backprop: bool = True    # Whether to compute gradients
) -> dict
```

Returns: `prediction`, `layers` (activations), `dynamicEdges`, `topContributors`, `backprop`, `weightStats`

##### Adjustments Parameter

```python
adjustments = {
    "layerScales": [1.0, 1.0, 1.0],        # Multiply weights per layer
    "outputBiasOffsets": [0.0, 0.0, ...],     # Add to output neurons
    "targetDigit": None                       # Force target for backprop
}
```

| Adjustment             | Default | Range     | Effect                           |
| ---------------------- | ------- | --------- | -------------------------------- |
| `layerScales[0]`       | 1.0     | 0.45-1.75 | Scale input→hidden1 weights      |
| `layerScales[1]`       | 1.0     | 0.45-1.75 | Scale hidden1→hidden2 weights    |
| `layerScales[2]`       | 1.0     | 0.45-1.75 | Scale hidden2→output weights     |
| `outputBiasOffsets[n]` | 0.0     | -2 to +2  | Add to output neuron n's value   |
| `targetDigit`          | auto    | 0-9       | Force specific prediction target |

#### FastAPI Endpoints

##### `POST /analyze`

Request:

```json
{
  "pixels": [[0.0, 0.5, ...], ...],  // 28x28 grid
  "adjustments": {
    "layerScales": [1.0, 1.0, 1.0],
    "outputBiasOffsets": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    "targetDigit": null
  },
  "includeBackprop": true
}
```

Response:

```json
{
  "prediction": {
    "digit": 5,
    "confidence": 0.9234,
    "probabilities": [0.001, 0.002, 0.001, 0.005, 0.002, 0.923, 0.001, 0.003, 0.050, 0.012]
  },
  "layers": [
    {"name": "input", "size": 784, "activations": [...], "stats": {...}},
    {"name": "hidden_1", "size": 96, "activations": [...], "stats": {...}},
    {"name": "hidden_2", "size": 48, "activations": [...], "stats": {...}},
    {"name": "output", "size": 10, "activations": [...], "stats": {...}}
  ],
  "dynamicEdges": {
    "input_to_hidden_1": [...],
    "hidden_1_to_hidden_2": [...],
    "hidden_2_to_output": [...]
  },
  "backprop": {
    "loss": 0.1234,
    "targetDigit": 5,
    "layerGradientNorms": [0.567, 0.234, 0.123, 0.089],
    "strongestWeightUpdates": {...}
  }
}
```

##### `POST /train-step`

Performs one step of backpropagation training:

```json
{
  "pixels": [[0.0, 0.5, ...], ...],
  "adjustments": {...},
  "label": 5  // The correct digit (0-9)
}
```

#### Frontend Canvas

##### `DigitCanvas.jsx` - Drawing Parameters

```javascript
const GRID_SIZE = 28; // MNIST standard
const CELL_SIZE = 12; // Rendered pixel size
const CANVAS_SIZE = 336; // Total dimension

// Brush parameters
brushRadius: 1 - 4; // Size of drawing brush
brushStrength: 0.4 - 1.5; // Drawing intensity
```

##### `digitGenerator.js` - Demo Digit Generation

```javascript
function buildHandwrittenDigit(digit)  // digit: 0-9
// Returns: 28x28 grid array

// Adds variation:
// - Random rotation: ±0.2 radians
// - Random scale: 0.8-1.1
// - Gaussian blur for realistic look
```

### 6. Data Flow Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         USER'S BROWSER                               │
│                                                                      │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────────────────┐ │
│  │  Drawing    │───▶│   React     │───▶│   Socket.IO Client      │ │
│  │  Canvas     │    │   App.jsx   │    │   (Real-time)           │ │
│  └─────────────┘    └─────────────┘    └───────────┬─────────────┘ │
│                                                       │              │
└───────────────────────────────────────────────────────┼──────────────┘
                                                        │
                           WebSocket Connection          │
                                                        ▼
┌──────────────────────────────────────────────────────────────────────┐
│                     EXPRESS SERVER (Node.js)                         │
│                      backend/server.js :4000                          │
│                                                                      │
│  ┌────────────────────────────────────────────────────────────────┐  │
│  │                     Socket.IO Handler                           │  │
│  │   • receive 'analyze-digit' event                              │  │
│  │   • forward to FastAPI via Axios                               │  │
│  │   • broadcast results to all clients                           │  │
│  └────────────────────────────┬───────────────────────────────────┘  │
│                               │                                       │
│                               │ HTTP POST                             │
│                               ▼                                       │
│  ┌────────────────────────────────────────────────────────────────┐  │
│  │                    REST API Endpoints                           │  │
│  │   /api/health  /api/model  /api/analyze  /api/train-step      │  │
│  └────────────────────────────┬───────────────────────────────────┘  │
│                               │                                       │
└───────────────────────────────┼───────────────────────────────────────┘
                                │
                                │ HTTP
                                ▼
┌──────────────────────────────────────────────────────────────────────┐
│                  FASTAPI SERVER (Python)                            │
│                  backend/ml_server.py :8000                         │
│                                                                      │
│  ┌────────────────────────────────────────────────────────────────┐  │
│  │                    ML Inference Engine                         │  │
│  │   • Load PyTorch model                                         │  │
│  │   • Forward pass through network                               │  │
│  │   • Compute gradients (optional)                               │  │
│  │   • Return activations, predictions, backprop data             │  │
│  └────────────────────────────┬───────────────────────────────────┘  │
│                               │                                       │
│                               ▼                                       │
│  ┌────────────────────────────────────────────────────────────────┐  │
│  │                    PyTorch Model                              │  │
│  │   backend/model_utils.py                                       │  │
│  │   • VisualizerNet (784→96→48→10)                             │  │
│  │   • Trained weights from mnist_visualizer_model.pth           │  │
│  └────────────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────────────┘
                                │
                                │ (optional)
                                ▼
┌──────────────────────────────────────────────────────────────────────┐
│                    MONGODB ATLAS (Cloud)                            │
│                 (Optional - falls back to memory)                    │
│                                                                      │
│  Database: NumberPredictor                                          │
│  Collection: predictionHistory                                      │
│  Stores: {id, createdAt, digit, confidence, probabilities, controls} │
└──────────────────────────────────────────────────────────────────────┘
```

### 7. Use Cases

#### For Faculty Demonstrations

1. **Live Drawing**: Draw any digit 0-9 on the canvas and see instant predictions
2. **Automated Showcase**: Click "Start Showcase" for step-by-step animated explanation
3. **Layer Manipulation**: Adjust layer scales to see how weights affect predictions
4. **Bias Override**: Modify output biases to understand neuron competition
5. **Backprop Visualization**: See gradients and weight updates in real-time

#### For Students Learning

1. **Architecture Understanding**: Visualize 784→96→48→10 network structure
2. **Activation Flow**: Watch signals propagate through layers
3. **Weight Visualization**: See heatmaps of learned weights
4. **Error Analysis**: Understand why the network makes mistakes

#### For Model Experimentation

1. **Live Training**: Apply single training steps with labeled data
2. **Sensitivity Analysis**: Test how small weight changes affect predictions
3. **Feature Analysis**: Identify which hidden neurons respond to certain features

### 8. Interactive Controls

| Control         | Location | Range      | Purpose                           |
| --------------- | -------- | ---------- | --------------------------------- |
| Brush Radius    | Canvas   | 1-4        | Size of drawing brush             |
| Brush Strength  | Canvas   | 0.4-1.5    | Drawing intensity                 |
| Layer Scale 1   | Controls | 0.45-1.75× | Input→Hidden1 weight multiplier   |
| Layer Scale 2   | Controls | 0.45-1.75× | Hidden1→Hidden2 weight multiplier |
| Layer Scale 3   | Controls | 0.45-1.75× | Hidden2→Output weight multiplier  |
| Bias Override   | Controls | -2 to +2   | Add to specific output neuron     |
| Target Digit    | Controls | 0-9        | Force backprop target             |
| Backprop Toggle | Controls | On/Off     | Enable gradient computation       |
| 3D Scene Toggle | Controls | On/Off     | Show/hide 3D visualization        |

### 9. Performance Notes

- **Model Accuracy**: 96.72% on MNIST test set
- **Inference Speed**: ~10ms per prediction
- **Frontend Bundle**: ~200KB (main) + ~400KB (Three.js, lazy-loaded)
- **Socket.IO**: Supports hundreds of concurrent connections
- **MongoDB Fallback**: Automatically uses in-memory storage if Atlas unavailable

### 10. Environment Variables

| Variable             | Default               | Description                     |
| -------------------- | --------------------- | ------------------------------- |
| `PORT`               | 4000                  | Express server port             |
| `PYTHON_SERVICE_URL` | http://127.0.0.1:8000 | FastAPI endpoint                |
| `MONGODB_URI`        | (not set)             | MongoDB Atlas connection string |
| `MONGODB_DB_NAME`    | NumberPredictor       | Database name                   |
| `MONGODB_COLLECTION` | predictionHistory     | Collection name                 |
| `MAX_HISTORY`        | 18                    | Maximum history entries         |
| `VITE_API_BASE_URL`  | (empty)               | Production API URL              |
| `VITE_SOCKET_URL`    | (empty)               | Production Socket URL           |
