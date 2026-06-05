import streamlit as st
import cv2
import tempfile
import torch
import torch.nn as nn
import numpy as np
import os
import pandas as pd
from ultralytics import YOLO

# System Configuration
INPUT_SIZE = 51
HIDDEN_SIZE = 128
NUM_LAYERS = 2
SEQUENCE_LENGTH = 90
TEMPERATURE = 0.5  # Softmax scaling factor for prediction confidence

class ActionLSTM(nn.Module):
    """ Inference network definition matching the trained artifact architecture. """
    def __init__(self, num_classes):
        super(ActionLSTM, self).__init__()
        self.lstm = nn.LSTM(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(HIDDEN_SIZE, num_classes)
        
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

def normalize_skeleton(frame_features):
    """ Applies the same spatial normalization logic used during the training phase. """
    x_idx = np.arange(0, 51, 3)
    y_idx = np.arange(1, 51, 3)
    conf_idx = np.arange(2, 51, 3)
    
    valid_mask = frame_features[conf_idx] > 0.2
    if np.any(valid_mask):
        mean_x = np.mean(frame_features[x_idx][valid_mask])
        mean_y = np.mean(frame_features[y_idx][valid_mask])
        frame_features[x_idx][valid_mask] = (frame_features[x_idx][valid_mask] - mean_x) / 640.0
        frame_features[y_idx][valid_mask] = (frame_features[y_idx][valid_mask] - mean_y) / 480.0
        
    return frame_features

@st.cache_resource
def load_models():
    """ Caches network weights into memory for optimized inference speed. """
    yolo_model = YOLO("yolov8n-pose.pt")
    classes = np.load('classes.npy', allow_pickle=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    lstm_model = ActionLSTM(len(classes)).to(device)
    lstm_model.load_state_dict(torch.load('action_lstm.pth', map_location=device))
    lstm_model.eval()
    
    return yolo_model, lstm_model, classes, device

# Dashboard Initialization
st.set_page_config(page_title="Activity Analyzer", page_icon="🏋️", layout="wide")
st.title("Intelligent Video Activity Analyzer")
st.markdown("### Integrated Pipeline: Pose Estimation & Temporal Sequence Classification")

yolo_model, lstm_model, classes, device = load_models()

# Sidebar Control Panel
st.sidebar.header("Inference Parameters")
smoothing_window = st.sidebar.slider("Temporal Filtering Window (Frames)", min_value=5, max_value=25, value=15, step=2)
min_confidence = st.sidebar.slider("Confidence Threshold", min_value=0.3, max_value=0.95, value=0.5)

uploaded_file = st.file_uploader("Upload Video Source", type=['mp4', 'mov', 'avi'])

if uploaded_file is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    tfile.write(uploaded_file.read())
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("#### Input Media Stream")
        st.video(tfile.name)
        
    if st.button("Execute Analysis Pipeline"):
        raw_predictions_log = []
        smoothed_predictions_log = []
        confidence_history = []
        time_axis = []
        
        rep_count = 0
        in_exercise_state = False
        
        with st.spinner("Extracting structural features & performing inference..."):
            cap = cv2.VideoCapture(tfile.name)
            fps = cap.get(cv2.CAP_PROP_FPS)
            if fps == 0 or np.isnan(fps): fps = 30
            
            frame_buffer = []
            frame_idx = 0
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # YOLO Extraction
                results = yolo_model(frame, verbose=False)
                if results[0].keypoints is not None and len(results[0].keypoints.data) > 0:
                    raw_kp = results[0].keypoints.data[0].cpu().numpy().flatten()
                else:
                    raw_kp = np.zeros(INPUT_SIZE)
                    
                normalized_kp = normalize_skeleton(raw_kp)
                frame_buffer.append(normalized_kp)
                
                # Maintain Sliding Window
                if len(frame_buffer) > SEQUENCE_LENGTH:
                    frame_buffer.pop(0)
                    
                # Sequence Inference execution
                if len(frame_buffer) == SEQUENCE_LENGTH:
                    sequence_tensor = torch.tensor(frame_buffer, dtype=torch.float32).unsqueeze(0).to(device)
                    
                    with torch.no_grad():
                        outputs = lstm_model(sequence_tensor)
                        scaled_outputs = outputs / TEMPERATURE
                        probs = torch.nn.functional.softmax(scaled_outputs, dim=1).cpu().numpy()[0]
                        pred_idx = np.argmax(probs)
                        conf = probs[pred_idx]
                        
                    raw_pred_class = classes[pred_idx]
                    raw_predictions_log.append(raw_pred_class)
                    
                    # Temporal Filtering Logic
                    recent_predictions = raw_predictions_log[-smoothing_window:]
                    smoothed_pred_class = max(set(recent_predictions), key=recent_predictions.count)
                    
                    if conf < min_confidence:
                        smoothed_pred_class = "uncertain"
                        
                    smoothed_predictions_log.append(smoothed_pred_class)
                    confidence_history.append(conf)
                    time_axis.append(frame_idx / fps)
                    
                    # State-based repetition counter logic
                    if smoothed_pred_class in ['jump', 'pushup', 'situp']:
                        if not in_exercise_state and conf > 0.65:
                            in_exercise_state = True
                    else:
                        if in_exercise_state:
                            rep_count += 1
                            in_exercise_state = False
                            
                frame_idx += 1
            cap.release()
            
        with col2:
            st.markdown("#### Engine Diagnostics")
            
            m1, m2 = st.columns(2)
            
            # Calculate the majority classification across the ENTIRE video, 
            # ignoring the resting/uncertain frames at the beginning and end.
            valid_predictions = [p for p in smoothed_predictions_log if p != "uncertain"]
            
            if valid_predictions:
                final_pred = max(set(valid_predictions), key=valid_predictions.count)
            else:
                final_pred = "UNCERTAIN"
            
            m1.metric(label="Primary Classification", value=final_pred.upper())
            m2.metric(label="Detected Cycles (Reps)", value=f"{rep_count}")
            
            st.markdown("#### Confidence Telemetry Timeline")
            if smoothed_predictions_log:
                timeline_df = pd.DataFrame({
                    'Timestamp (s)': time_axis,
                    'Activity State': smoothed_predictions_log,
                    'Model Confidence': confidence_history
                })
                
                st.line_chart(data=timeline_df, x='Timestamp (s)', y='Model Confidence')
                with st.expander("View Raw Telemetry Data"):
                    st.dataframe(timeline_df, use_container_width=True)
            else:
                st.info("Input sequence must exceed the sliding window capacity (3 seconds).")
                
    os.unlink(tfile.name)