import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Model Hyperparameters
FEATURES_DIR = "balanced_features"
INPUT_SIZE = 51
HIDDEN_SIZE = 128
NUM_LAYERS = 2
BATCH_SIZE = 16
EPOCHS = 100
LEARNING_RATE = 0.0002

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Hardware utilized for training: {device}")

class ActivityDataset(Dataset):
    """ PyTorch Dataset wrapper for sequence tensors. """
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
    def __len__(self): 
        return len(self.y)
    def __getitem__(self, idx): 
        return self.X[idx], self.y[idx]

def load_preprocessed_data():
    """ Loads normalized sequence arrays from the preprocessed directory. """
    X, y = [], []
    activities = sorted([d for d in os.listdir(FEATURES_DIR) if os.path.isdir(os.path.join(FEATURES_DIR, d))])
    
    label_encoder = LabelEncoder()
    label_encoder.fit(activities)
    
    for activity in activities:
        activity_dir = os.path.join(FEATURES_DIR, activity)
        label = label_encoder.transform([activity])[0]
        
        for file in os.listdir(activity_dir):
            if file.endswith('.npy'):
                features = np.load(os.path.join(activity_dir, file))
                X.append(features)
                y.append(label)
                
    np.save('classes.npy', label_encoder.classes_)
    return np.array(X), np.array(y), label_encoder.classes_

class ActionLSTM(nn.Module):
    """ Temporal classification network utilizing Long Short-Term Memory architecture. """
    def __init__(self, num_classes):
        super(ActionLSTM, self).__init__()
        self.lstm = nn.LSTM(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(HIDDEN_SIZE, num_classes)
        
    def forward(self, x):
        out, _ = self.lstm(x)
        # Extract the sequence representation from the final time step
        return self.fc(out[:, -1, :])

def train_and_evaluate():
    """ Executes the end-to-end training loop and generates evaluation metrics. """
    print("Initializing data loaders...")
    X, y, classes = load_preprocessed_data()
    print(f"Ingested {len(X)} sequence samples across {len(classes)} classes.")
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    train_loader = DataLoader(ActivityDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(ActivityDataset(X_test, y_test), batch_size=BATCH_SIZE, shuffle=False)
    
    model = ActionLSTM(len(classes)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    print("\nInitiating training sequence...")
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(inputs.to(device)), labels.to(device))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        if (epoch + 1) % 5 == 0:
            print(f"Epoch [{epoch + 1}/{EPOCHS}] | Training Loss: {total_loss/len(train_loader):.4f}")
            
    print("\nRunning inference on test dataset...")
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs.to(device))
            all_preds.extend(torch.max(outputs, 1)[1].cpu().numpy())
            all_labels.extend(labels.numpy())
            
    print("\n--- Model Evaluation Metrics ---")
    print(classification_report(all_labels, all_preds, target_names=classes))
    
    # Generate and export confusion matrix visualization
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title('Validation Confusion Matrix')
    plt.ylabel('Ground Truth')
    plt.xlabel('Predicted Observation')
    plt.savefig('confusion_matrix.png')
    
    # Export trained weights
    torch.save(model.state_dict(), 'action_lstm.pth')
    print("Training sequence completed. Artifacts saved: 'action_lstm.pth', 'confusion_matrix.png'")

if __name__ == "__main__":
    train_and_evaluate()