import torch
import torch.nn as nn
from torchvision.models import resnet50
class DriverBehaviorClassifier(nn.Module):
    def __init__(self, num_classes=10):
        super(DriverBehaviorClassifier, self).__init__()
        self.resnet = resnet50(weights=None)
        self.resnet.fc = nn.Identity()  # 移除最后一层
        self.fc1 = nn.Linear(2048 + 133 * 2, 512)  # 2048 是 ResNet50 的输出维度，133 * 2 是关键点和分数的总维度
        self.fc2 = nn.Linear(512, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, image, keypoints, keypoint_scores):
        global_features = self.resnet(image)
        keypoints = keypoints.view(keypoints.size(0), -1)
        keypoint_scores = keypoint_scores.view(keypoint_scores.size(0), -1)
        combined_features = torch.cat((global_features, keypoints, keypoint_scores), dim=1)
        x = self.relu(self.fc1(combined_features))
        x = self.dropout(x)
        x = self.fc2(x)
        return x