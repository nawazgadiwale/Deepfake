# model/dataset.py
import os
import cv2
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

class DeepfakeDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.data = []

        for label_name, label in [('real', 0), ('fake', 1)]:
            class_dir = os.path.join(root_dir, label_name)
            if not os.path.exists(class_dir):
                continue
            for img in os.listdir(class_dir):
                if img.lower().endswith(('.jpg', '.jpeg', '.png')):
                    self.data.append((os.path.join(class_dir, img), label))

        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        path, label = self.data[idx]
        image = cv2.imread(path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return self.transform(image), label
