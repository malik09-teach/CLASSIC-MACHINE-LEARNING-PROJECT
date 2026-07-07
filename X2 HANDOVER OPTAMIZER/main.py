from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import pickle
import pandas as pd
import uvicorn

app = FastAPI()
templates = Jinja2Templates(directory="templates")

# ── 1. Load the ML Engine and Data ──
try:
    with open('towerguard_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('towerguard_preprocessor.pkl', 'rb') as f:
        preprocessor = pickle.load(f)
    stream_data = pd.read_csv('live_stream_data.csv')
    print("✅ AI Engine Loaded Successfully")
except Exception as e:
    print(f"🚨 Error loading files: {e}")

# Global counter to track our live stream position
current_row_idx = 0

@app.get("/", response_class=HTMLResponse)
async def serve_ui(request: Request):
    """Serves your custom HTML dashboard."""
    return templates.TemplateResponse(request=request, name="index.html")

@app.get("/reset")
def reset_stream():
    """Resets the data stream when the user clicks restart."""
    global current_row_idx
    current_row_idx = 0
    return {"status": "success"}

@app.get("/next_tick")
def get_next_prediction():
    """Processes the next row of data through the ML model."""
    global current_row_idx
    
    if current_row_idx >= len(stream_data):
        return {"status": "end_of_stream"}
        
    # Grab the current row
    row = stream_data.iloc[current_row_idx]
    single_row_df = row.to_frame().T
    
    # Preprocess and Predict using your Pickle models
    processed_data = preprocessor.transform(single_row_df)
    prediction = int(model.predict(processed_data)[0])
    confidence = float(model.predict_proba(processed_data)[0][1])
    
    # Extract metrics
    current_load = float(row['resource_block_util'] * 100)
    
    # Move to the next row for the next API call
    current_row_idx += 1
    
    return {
        "status": "active",
        "load": current_load,
        "prediction": prediction,      # 0 (Normal) or 1 (Crash Imminent)
        "confidence": confidence       # e.g., 0.85 (85% sure)
    }

if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)