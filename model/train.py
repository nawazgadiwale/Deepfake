import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import DeepfakeDataset
from detector import DeepfakeDetector
from utils import save_model
from sklearn.metrics import accuracy_score
from tqdm import tqdm

MODEL_PATH = "../saved_models/deepfake_detector.pth"

def train():
    # Skip training if model already exists
    if os.path.exists(MODEL_PATH):
        print(f"✅ Model already exists at {MODEL_PATH}. Skipping training.")
        return

    print("🚀 Starting training...")
    
    # Load dataset
    dataset = DeepfakeDataset("../data/images")
    loader = DataLoader(dataset, batch_size=16, shuffle=True)

    # Initialize model
    model = DeepfakeDetector()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Loss function: BCEWithLogitsLoss is used for raw logits
    criterion = nn.BCEWithLogitsLoss()  # Using BCEWithLogitsLoss for raw logits
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # Train for only 1 epoch
    model.train()
    losses, y_true, y_pred = [], [], []

    print(f"\n🌀 Epoch 1 starting...")
    for images, labels in tqdm(loader, desc="Training Batches"):
        images = images.to(device)
        labels = labels.float().unsqueeze(1).to(device)

        optimizer.zero_grad()
        outputs = model(images)
        
        # Calculate loss
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        y_true += labels.cpu().numpy().tolist()
        y_pred += (torch.sigmoid(outputs).cpu().detach().numpy() > 0.5).astype(int).tolist()  # Sigmoid on output

    acc = accuracy_score(y_true, y_pred)
    print(f"\n✅ Training Complete → Loss: {sum(losses)/len(losses):.4f}, Accuracy: {acc:.4f}")

    # Save the model
    save_model(model, MODEL_PATH)
    print(f"\n💾 Model saved at {MODEL_PATH}")

if __name__ == "__main__":
    train()
