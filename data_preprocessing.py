# D:/PointCloud Manifold/now/data/ModelNet40

import os
import numpy as np
import trimesh
from sklearn.model_selection import train_test_split

def load_point_cloud(file_path, max_points=1024):
    mesh = trimesh.load(file_path)
    points = mesh.vertices
    num_points = points.shape[0]

    if num_points > max_points:
        # Truncate the point cloud if it has more points than the max_points
        points = points[:max_points, :]
    elif num_points < max_points:
        # Pad the point cloud with zeros if it has fewer points than the max_points
        pad_points = np.zeros((max_points - num_points, 3), dtype=np.float32)
        points = np.concatenate((points, pad_points), axis=0)

    return points

def preprocess_dataset(data_dir, max_points=1024, test_split=0.2, random_state=42):
    categories = os.listdir(data_dir)
    X_train, X_test, y_train, y_test = [], [], [], []

    for i, category in enumerate(categories):
        category_dir = os.path.join(data_dir, category)
        if not os.path.isdir(category_dir):
            continue

        train_dir = os.path.join(category_dir, "train")
        test_dir = os.path.join(category_dir, "test")

        if not os.path.isdir(train_dir) or not os.path.isdir(test_dir):
            continue

        train_files = [os.path.join(train_dir, f) for f in os.listdir(train_dir) if f.endswith(".off")]
        test_files = [os.path.join(test_dir, f) for f in os.listdir(test_dir) if f.endswith(".off")]

        train_points = [load_point_cloud(file, max_points).transpose() for file in train_files]
        test_points = [load_point_cloud(file, max_points).transpose() for file in test_files]

        # Split data into train and test sets
        X_train.extend(train_points)
        X_test.extend(test_points)
        y_train.extend([i] * len(train_points))
        y_test.extend([i] * len(test_points))

    return np.array(X_train), np.array(X_test), np.array(y_train), np.array(y_test)

if __name__ == "__main__":
    data_dir = "D:/PointCloud Manifold/now/data/ModelNet40"
    X_train, X_test, y_train, y_test = preprocess_dataset(data_dir)
    np.save("data/X_train.npy", X_train)
    np.save("data/X_test.npy", X_test)
    np.save("data/y_train.npy", y_train)
    np.save("data/y_test.npy", y_test)

