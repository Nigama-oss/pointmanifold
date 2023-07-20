# evaluate_point_manifold.py

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from point_manifold_model import PointManifoldModel

def evaluate_model(X_test, y_test):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.long)

    test_dataset = TensorDataset(X_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    model = PointManifoldModel(num_classes).to(device)
    model.load_state_dict(torch.load("point_manifold_model.pt"))
    model.eval()

    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100.0 * correct / total
    print(f"Test Accuracy: {accuracy:.2f}%")

if __name__ == "__main__":
    X_test = np.load("data/X_test.npy")
    y_test = np.load("data/y_test.npy")

    num_classes = len(np.unique(y_test))
    evaluate_model(X_test, y_test)
