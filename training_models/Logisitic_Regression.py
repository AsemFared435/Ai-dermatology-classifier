from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch
import os
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

train_dataset = datasets.ImageFolder('../data/processed/train', transform=transform)
val_dataset = datasets.ImageFolder('../data/processed/val', transform=transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

def extract_features(loader):
    features = []
    labels = []
    for imgs, lbls in loader:
        imgs = imgs.view(imgs.size(0), -1).numpy()  # Flatten
        features.extend(imgs)
        labels.extend(lbls.numpy())
    return np.array(features), np.array(labels)

X_train, y_train = extract_features(train_loader)
X_val, y_val = extract_features(val_loader)

model = LogisticRegression(max_iter=1000, multi_class='multinomial')
model.fit(X_train, y_train)

val_acc = accuracy_score(y_val, model.predict(X_val))
print(f"Logistic Regression Validation Accuracy: {val_acc:.4f}")

import joblib
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SAVE_DIR = os.path.join(BASE_DIR, "..", "Saved_models")
os.makedirs(SAVE_DIR, exist_ok=True)
MODEL_PATH = os.path.join(SAVE_DIR, "efficientnet_best.pth")
joblib.dump(model, MODEL_PATH)
print("✅ Logistic Regression model saved successfully!")