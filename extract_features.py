import cv2
import numpy as np
import os
import torch
from ultralytics import YOLO

# Configuration
DATASET_DIR = "dataset"
OUTPUT_DIR = "extracted_features"
SEQUENCE_LENGTH = 90  # Target frames per sequence

# Initialize YOLOv8 Nano Pose Model
print("Loading YOLOv8 Pose Model...")
model = YOLO("yolov8n-pose.pt")

def extract_features_from_frames():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 1. Find the activity folders (e.g., jump, squat)
    activities = [d for d in os.listdir(DATASET_DIR) if os.path.isdir(os.path.join(DATASET_DIR, d))]
    print(f"Found activities: {activities}")

    for activity in activities:
        activity_dir = os.path.join(DATASET_DIR, activity)
        out_activity_dir = os.path.join(OUTPUT_DIR, activity)
        os.makedirs(out_activity_dir, exist_ok=True)
        
        # 2. Find the individual sequence/clip folders inside each activity
        sequences = [d for d in os.listdir(activity_dir) if os.path.isdir(os.path.join(activity_dir, d))]
        print(f"\nProcessing {len(sequences)} frame sequences for '{activity}'...")
        
        for seq in sequences:
            seq_dir = os.path.join(activity_dir, seq)
            out_file = os.path.join(out_activity_dir, seq + '.npy')
            
            if os.path.exists(out_file):
                continue
                
            # 3. Grab all frames in the folder and sort them numerically/alphabetically
            # This ensures frame_01 comes before frame_02
            valid_extensions = ('.jpg', '.jpeg', '.png')
            frames = sorted([f for f in os.listdir(seq_dir) if f.lower().endswith(valid_extensions)])
            
            if not frames:
                print(f"Warning: No images found in {seq_dir}")
                continue
                
            video_features = []
            
            # 4. Process up to SEQUENCE_LENGTH frames
            for i in range(SEQUENCE_LENGTH):
                if i < len(frames):
                    frame_path = os.path.join(seq_dir, frames[i])
                    image = cv2.imread(frame_path)
                    
                    if image is None:
                        video_features.append(np.zeros(51))
                        continue
                        
                    # Run YOLO inference
                    results = model(image, verbose=False)
                    
                    # Check for human keypoints
                    if results[0].keypoints is not None and len(results[0].keypoints.data) > 0:
                        keypoints = results[0].keypoints.data[0].cpu().numpy().flatten()
                    else:
                        keypoints = np.zeros(51)
                else:
                    # If the folder has fewer than 90 frames, pad the rest with zeros
                    keypoints = np.zeros(51)
                    
                video_features.append(keypoints)
                
            # Save the sequence as a lightweight .npy array
            np.save(out_file, np.array(video_features))
            
    print("\n✅ YOLOv8 Feature extraction complete! Data saved as .npy files.")

if __name__ == "__main__":
    extract_features_from_frames()