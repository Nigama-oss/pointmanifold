import torch
import torch.nn as nn

class PointManifoldModel(nn.Module):
    def __init__(self, num_classes):
        super(PointManifoldModel, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=3, out_channels=64, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=1)
        self.conv3 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=1)
        self.conv4 = nn.Conv1d(in_channels=256, out_channels=512, kernel_size=1)
        self.conv5 = nn.Conv1d(in_channels=512, out_channels=1024, kernel_size=1)

        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_classes)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))
        x = self.relu(self.conv5(x))

        x = torch.max(x, dim=2)[0]  # Global max pooling

        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.fc3(x)

        return x
