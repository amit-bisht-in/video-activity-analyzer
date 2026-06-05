import os
import numpy as np
import shutil

# Hyperparameters and Paths
INPUT_DIR = "extracted_features"
OUTPUT_DIR = "balanced_features"
SEQUENCE_LENGTH = 90
INPUT_SIZE = 51

def normalize_skeleton(features):
    """
    Centers spatial coordinates around the frame mean and applies scaling.
    Ensures model learns positional invariance rather than exact pixel locations.
    """
    normalized = np.copy(features)
    for i in range(len(normalized)):
        frame = normalized[i]
        
        # YOLOv8 keypoint format: [x1, y1, conf1, x2, y2, conf2, ...]
        x_idx = np.arange(0, 51, 3)
        y_idx = np.arange(1, 51, 3)
        conf_idx = np.arange(2, 51, 3)
        
        # Filter for keypoints with acceptable tracking confidence
        valid_mask = frame[conf_idx] > 0.2
        
        if np.any(valid_mask):
            mean_x = np.mean(frame[x_idx][valid_mask])
            mean_y = np.mean(frame[y_idx][valid_mask])
            
            frame[x_idx][valid_mask] = (frame[x_idx][valid_mask] - mean_x) / 640.0
            frame[y_idx][valid_mask] = (frame[y_idx][valid_mask] - mean_y) / 480.0
            
        normalized[i] = frame
    return normalized

def augment_skeleton(features):
    """
    Applying synthetic augmentation via horizontal mirroring and Gaussian jitter
    to prevent overfitting on minority classes during upsampling.
    """
    augmented = np.copy(features)
    x_idx = np.arange(0, 51, 3)
    y_idx = np.arange(1, 51, 3)
    
    # Random horizontal flip probability (50%)
    if np.random.rand() > 0.5:
        augmented[:, x_idx] = augmented[:, x_idx] * -1 
        
    # Inject Gaussian noise to simulate camera displacement
    noise = np.random.normal(0, 0.02, augmented.shape)
    augmented[:, x_idx] += noise[:, x_idx]
    augmented[:, y_idx] += noise[:, y_idx]
    
    return augmented

def process_and_balance_data():
    """
    Main preprocessing: standardizes sequence lengths, applies spatial
    normalization, and balances class distribution via augmented oversampling.
    """
    if os.path.exists(OUTPUT_DIR):
        print(f"Cleaning existing directory: {OUTPUT_DIR}...")
        shutil.rmtree(OUTPUT_DIR)
    os.makedirs(OUTPUT_DIR)

    activities = sorted([d for d in os.listdir(INPUT_DIR) if os.path.isdir(os.path.join(INPUT_DIR, d))])
    
    # Determine target volume for class balancing
    class_counts = {}
    for activity in activities:
        files = [f for f in os.listdir(os.path.join(INPUT_DIR, activity)) if f.endswith('.npy')]
        class_counts[activity] = len(files)
        
    max_size = max(class_counts.values())
    print(f"Dataset Distribution: {class_counts}")
    print(f"Target upsample size: {max_size} samples per class\n")

    for activity in activities:
        print(f"Processing sequence data for class: '{activity}'")
        input_act_dir = os.path.join(INPUT_DIR, activity)
        output_act_dir = os.path.join(OUTPUT_DIR, activity)
        os.makedirs(output_act_dir, exist_ok=True)
        
        files = [f for f in os.listdir(input_act_dir) if f.endswith('.npy')]
        processed_sequences = []
        
        # Phase 1: Process and normalize original dataset
        for idx, file in enumerate(files):
            features = np.load(os.path.join(input_act_dir, file))
            
            # Standardize temporal length
            if len(features) > SEQUENCE_LENGTH:
                features = features[:SEQUENCE_LENGTH]
            elif len(features) < SEQUENCE_LENGTH:
                # Pad by repeating the final frame state to maintain temporal consistency
                deficit = SEQUENCE_LENGTH - len(features)
                last_frame = features[-1] if len(features) > 0 else np.zeros(INPUT_SIZE)
                padding = np.tile(last_frame, (deficit, 1))
                features = np.vstack((features, padding))
                
            normalized_features = normalize_skeleton(features)
            processed_sequences.append(normalized_features)
            np.save(os.path.join(output_act_dir, f"orig_{idx}.npy"), normalized_features)
            
        # Phase 2: Upsample minority classes via augmentation
        deficit = max_size - len(files)
        if deficit > 0:
            for idx in range(deficit):
                base_sequence = processed_sequences[np.random.randint(0, len(processed_sequences))]
                synthetic_sequence = augment_skeleton(base_sequence)
                np.save(os.path.join(output_act_dir, f"synth_{idx}.npy"), synthetic_sequence)
                
    print(f"\nPreprocessing Complete. Generated {max_size * len(activities)} normalized sequences.")

if __name__ == "__main__":
    process_and_balance_data()