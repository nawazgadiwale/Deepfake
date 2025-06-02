# video_processing.py

import cv2
import torch
from torchvision import transforms
import numpy as np
from model import DeepfakeDetector  # Import your model definition

# Preprocessing pipeline for the frames
preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Adjust based on your data
])

def extract_frames_from_video(video_path):
    """Extract frames from a video file."""
    cap = cv2.VideoCapture(video_path)
    frames = []
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert to RGB
        frames.append(frame)

    cap.release()
    return frames

def predict_frame(frame, model, device):
    """Predict if the frame is real or fake."""
    frame = preprocess(frame)
    frame = frame.unsqueeze(0).to(device)  # Add batch dimension

    with torch.no_grad():
        output = model(frame)
        prediction = torch.sigmoid(output).item()  # Probability (0 = real, 1 = fake)

    return prediction

def predict_video(video_path, model, device, threshold=0.5):
    """Predict if the video is real or fake."""
    frames = extract_frames_from_video(video_path)
    frame_predictions = []

    for frame in frames:
        prediction = predict_frame(frame, model, device)
        frame_predictions.append(prediction)

    avg_confidence = np.mean(frame_predictions)
    print(f"Average Frame Confidence: {avg_confidence}")

    # Final classification decision
    if avg_confidence > threshold:
        return "Fake"
    else:
        return "Real"
