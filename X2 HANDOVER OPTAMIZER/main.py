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

# Global counter and mitigation state
current_row_idx = 0
mitigation_offset = 0.0
is_mitigating = False

@app.get("/", response_class=HTMLResponse)
async def serve_ui(request: Request):
    """Serves your custom HTML dashboard."""
    return templates.TemplateResponse(request=request, name="index.html")

@app.get("/reset")
def reset_stream():
    """Resets the data stream when the user clicks restart."""
    global current_row_idx, mitigation_offset, is_mitigating
    current_row_idx = 0
    mitigation_offset = 0.0
    is_mitigating = False
    return {"status": "success"}

@app.get("/next_tick")
def get_next_prediction(ai_enabled: bool = False, manual_load: float = 0.0):
    """Processes the next row of data through the ML model."""
    global current_row_idx, mitigation_offset, is_mitigating
    
    if current_row_idx >= len(stream_data):
        return {"status": "end_of_stream"}
        
    # Grab the current row and copy to allow modification
    row = stream_data.iloc[current_row_idx].copy()
    
    # Inject manual traffic boost BEFORE preprocessing so the ML model sees it
    if manual_load > 0:
        row['resource_block_util'] += (manual_load / 100.0)
        if 'active_users' in row:
            row['active_users'] += int(manual_load * 5)
            
    single_row_df = row.to_frame().T
    
    # Preprocess and Predict using your Pickle models
    processed_data = preprocessor.transform(single_row_df)
    prediction = int(model.predict(processed_data)[0])
    confidence = float(model.predict_proba(processed_data)[0][1])
    
    # Extract base load
    base_load = float(row['resource_block_util'] * 100)
    
    # Handle mitigation dynamics
    if ai_enabled and prediction == 1 and not is_mitigating:
        is_mitigating = True
        
    if is_mitigating:
        # Gradually increase offloaded amount up to ~55% reduction
        if mitigation_offset < 55.0:
            mitigation_offset += 12.0 # Mitigation speed
        if mitigation_offset > 55.0:
            mitigation_offset = 55.0
            
    current_load = max(0.0, base_load - mitigation_offset)
    
    # Check if crashed
    is_crashed = False
    if current_load >= 95.0:
        is_crashed = True
    
    # Move to the next row for the next API call
    current_row_idx += 1
    
    return {
        "status": "active",
        "load": current_load,
        "prediction": prediction,      # 0 (Normal) or 1 (Crash Imminent)
        "confidence": confidence,      # e.g., 0.85 (85% sure)
        "is_mitigating": is_mitigating,
        "is_crashed": is_crashed
    }

if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)