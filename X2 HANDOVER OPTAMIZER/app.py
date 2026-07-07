import streamlit as st
import pickle
import pandas as pd
import time

# ── 1. Page Configuration ──
st.set_page_config(page_title="TowerGuard Live SON", layout="wide", page_icon="📡")
st.title(" TowerGuard: Live Edge Inference Dashboard")
st.markdown("Compare traditional reactive networks vs. AI-driven predictive networks.")
st.divider()

# ── 2. Load the Engine & Data ──
@st.cache_resource
def load_engine():
    try:
        with open('towerguard_model.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('towerguard_preprocessor.pkl', 'rb') as f:
            prep = pickle.load(f)
        data = pd.read_csv('live_stream_data.csv')
        return model, prep, data
    except Exception as e:
        return None, None, None

model, preprocessor, stream_data = load_engine()

if model is None:
    st.error(" Missing .pkl or .csv files! Please run the export code in your Jupyter Notebook.")
    st.stop()

# ── 3. Interactive Controls ──
col_ctrl1, col_ctrl2 = st.columns([1, 2])
with col_ctrl1:
    ai_enabled = st.toggle(" **Enable AI Optimizer Engine**", value=True)
with col_ctrl2:
    start_button = st.button(" Start Traffic Jam", use_container_width=True)

st.divider()

# ── 4. UI Metrics Layout ──
col1, col2, col3, col4 = st.columns(4)
metric_load = col1.empty()
metric_users = col2.empty()
metric_conf = col3.empty()
metric_status = col4.empty()

st.subheader("Network Load Trajectory")
chart_placeholder = st.empty()
action_alert = st.empty()

# ── 5. The Real-Time Stream Loop ──
if start_button:
    load_history = []
    
    # Initialize metrics at 0
    metric_load.metric("Tower Load", "0.0%")
    metric_users.metric("Active Users", "0")
    metric_conf.metric("AI Crash Prob.", "0.0%")
    metric_status.metric("Network Status", " IDLE")
    
    for idx, row in stream_data.iterrows():
        # Preprocess the data
        single_row_df = row.to_frame().T
        processed_data = preprocessor.transform(single_row_df)
        
        # AI Inference
        prediction = model.predict(processed_data)[0]
        confidence = model.predict_proba(processed_data)[0][1]
        
        # Extract live values
        current_load = row['resource_block_util'] * 100
        users = int(row['active_users'])
        load_history.append(current_load)
        
        # Update UI Metrics
        metric_load.metric("Tower Load", f"{current_load:.1f}%")
        metric_users.metric("Active Users", f"{users:,}")
        metric_conf.metric("AI Crash Prob.", f"{confidence:.1%}")
        
        # Update Chart
        chart_placeholder.line_chart(load_history, height=300)
        
        # ── ROUTING LOGIC ──
        
        # SCENARIO A: The Tower actually hits max capacity (Engine was OFF or failed)
        if current_load >= 99.0:
            metric_status.metric("Network Status", " CRASHED")
            action_alert.error(" HARDWARE CRASH! The tower reached 100% capacity. All calls dropped.")
            break # Simulation ends in failure
            
        # SCENARIO B: AI Predicts a crash and the Engine is ON
        elif prediction == 1 and ai_enabled:
            metric_status.metric("Network Status", " INTERVENING")
            action_alert.success(" AI DETECTED IMMINENT CRASH! Executing Automated X2 Handover...")
            
            # Simulate the load dropping instantly because users were moved
            time.sleep(0.8)
            for drop in range(1, 4): # Visually drop the line chart
                load_history.append(current_load - (drop * 15))
                chart_placeholder.line_chart(load_history, height=300)
                time.sleep(0.2)
                
            action_alert.success("✅ Handover complete. Network stabilized before crash occurred.")
            metric_load.metric("Tower Load", f"{load_history[-1]:.1f}%")
            metric_status.metric("Network Status", "🟢 SAVED")
            break # Simulation ends in success
            
        # SCENARIO C: AI Predicts a crash, but Engine is OFF
        elif prediction == 1 and not ai_enabled:
            metric_status.metric("Network Status", "🔴 CRITICAL")
            action_alert.warning(" AI is predicting a crash, but the Optimizer Engine is OFF. Taking no action.")
            
        # SCENARIO D: Normal Operation
        elif confidence > 0.30:
            metric_status.metric("Network Status", "🟡 WARNING")
            action_alert.info("Traffic building up. Monitoring trajectory...")
        else:
            metric_status.metric("Network Status", "🟢 STABLE")
            action_alert.empty()
            
        # Hardware tick delay (adjust to make graph faster/slower)
        time.sleep(0.4)

# ── 3. UI Layout ──
col1, col2, col3, col4 = st.columns(4)
metric_load = col1.empty()
metric_users = col2.empty()
metric_conf = col3.empty()
metric_status = col4.empty()

st.subheader("Network Load Trajectory")
chart_placeholder = st.empty()
action_alert = st.empty()

st.divider()
start_button = st.button("Start Live Network Stream", use_container_width=True)

# ── 4. The Real-Time Stream Loop ──
if start_button:
    load_history = []
    
    # Initialize metrics at 0
    metric_load.metric("Tower Load", "0.0%")
    metric_users.metric("Active Users", "0")
    metric_conf.metric("Crash Probability", "0.0%")
    metric_status.metric("Network Status", " IDLE")
    
    for idx, row in stream_data.iterrows():
        # Convert row to DataFrame for the preprocessor
        single_row_df = row.to_frame().T
        
        # Preprocess the data using your saved StandardScaler
        processed_data = preprocessor.transform(single_row_df)
        
        # AI Inference
        prediction = model.predict(processed_data)[0]
        confidence = model.predict_proba(processed_data)[0][1]
        
        # Extract live values
        current_load = row['resource_block_util'] * 100
        users = int(row['active_users'])
        load_history.append(current_load)
        
        # Update UI Metrics
        metric_load.metric("Tower Load", f"{current_load:.1f}%")
        metric_users.metric("Active Users", f"{users:,}")
        metric_conf.metric("Crash Probability", f"{confidence:.1%}")
        
        # Update Chart
        chart_placeholder.line_chart(load_history, height=300)
        
        # Action Logic
        if prediction == 1:
            metric_status.metric("Network Status", "🔴 CRITICAL")
            action_alert.error("🚨 CRASH IMMINENT: Executing Automated X2 Handover to Backup Tower...")
            time.sleep(1.5) # Pause to simulate the handover duration
            st.success(" Handover complete. Network stabilized. Disaster averted.")
            break # Stop stream after saving the tower
            
        elif confidence > 0.30:
            metric_status.metric("Network Status", "🟡 WARNING")
            action_alert.warning(" High network momentum detected. Priming spatial KNN grid...")
            
        else:
            metric_status.metric("Network Status", "🟢 STABLE")
            action_alert.info("Network operating within normal parameters.")
            
        # Hardware tick delay (adjust this to make the graph draw faster or slower)
        time.sleep(0.4)