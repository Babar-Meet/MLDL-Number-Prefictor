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
