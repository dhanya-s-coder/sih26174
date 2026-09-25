# sih26174
# HAR Space - On-board BAS Experiments AI

## Setup

1. Create a virtual environment:
   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On Mac/Linux:
   source venv/bin/activate
   ```
2. Install requirements:
   ```bash
   pip install -r requirements.txt
   ```

## Running Tests

Run `pytest` to execute all test cases:
```bash
pytest
```

## Running the Local Web Dashboard

The FastAPI process owns the single existing HAR pipeline (camera, perception, interaction tracking, TTS, logging, and processed stream). The React frontend is a separate presentation-only process. Start both from the project root in separate terminals.

Backend (Windows PowerShell):

```powershell
python -m pip install -r requirements.txt
python -m src.har_space.api.server --source 0
```

Start the React/Vite frontend:

```powershell
cd dashboard
npm install
npm run dev
```

Open the local Vite URL printed by npm (normally `http://127.0.0.1:5173`). The web UI receives runtime updates from the local API/WebSocket and displays the processed MJPEG stream produced by that same HAR process. It never accesses the webcam itself. Runtime communication and assets stay local; install Python/npm packages before offline operation.

Choose a different source or protocol with backend options, for example:

```powershell
python -m src.har_space.api.server --source path\to\experiment.mp4 --experiment configs\experiment_sample.yaml --mute
```

API endpoints: `/api/status`, `/api/experiment`, `/api/progress`, `/api/logs`, `/api/state`, and `/ws`. Run the OpenCV experiment view with `python scripts/run_demo.py` from the project root. Press `R` to reset without restarting the camera, or `Q`/`Esc` to quit. The API-backed runtime remains available separately through `python -m src.har_space.api.server`.

## Running Tests

```powershell
pytest
```
