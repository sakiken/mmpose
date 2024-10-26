import torch
import torch.nn as nn
from torchvision.models import resnet50
import torch.nn.functional as F
class DriverBehaviorClassifier(nn.Module):
    def __init__(self, num_classes=10):
        super(DriverBehaviorClassifier, self).__init__()
        self.resnet = resnet50(weights=None)
        self.resnet.fc = nn.Identity()  # 移除最后一层
        self.fc1 = nn.Linear(2048 + 133 * 2 + 133, 512)  # 2048 是 ResNet50 的输出维度，133 * 2 是关键点的总维度，133 是关键点分数的维度
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

class ChannelAttention(nn.Module):
    def __init__(self, num_channels, reduction_ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(num_channels, num_channels // reduction_ratio, 1, bias=False)
        self.relu = nn.ReLU()
        self.fc2 = nn.Conv2d(num_channels // reduction_ratio, num_channels, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)

class KeypointAttention(nn.Module):
    def __init__(self, num_keypoints):
        super(KeypointAttention, self).__init__()
        self.attention_weights = nn.Parameter(torch.randn(num_keypoints, 1))

    def forward(self, keypoints, scores):
        attention_weights = F.softmax(self.attention_weights, dim=0)
        # 将关键点的坐标和置信度分数与注意力权重相乘
        weighted_keypoints = keypoints * attention_weights.view(1, -1, 1)
        weighted_scores = scores * attention_weights.view(1, -1, 1)
        # 合并加权后的关键点坐标和置信度分数
        combined_keypoints = torch.cat((weighted_keypoints, weighted_scores), dim=2)
        return combined_keypoints.view(combined_keypoints.size(0), -1)

class DriverBehaviorClassifierWithAttention(nn.Module):
    def __init__(self, num_classes=10, num_keypoints=133):
        super(DriverBehaviorClassifierWithAttention, self).__init__()

        # 图像路径
        self.resnet = resnet50(weights=None)
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Identity()  # 移除最后的全连接层
        self.image_fc = nn.Linear(num_features, 512)
        self.image_classification_fc = nn.Linear(512, num_classes)  # 图像分类层

        # 关键点路径
        self.keypoint_fc = nn.Linear(num_keypoints * 3, 512)  # x, y, score
        self.keypoint_classification_fc = nn.Linear(512, num_classes)  # 关键点分类层

        # 注意力模块
        self.channel_attention = ChannelAttention(512)  # 通道注意力模块
        self.keypoint_attention = KeypointAttention(num_keypoints)

        self.bn = nn.BatchNorm1d(512)
        self.dropout = nn.Dropout(0.5)

    def forward(self, image, keypoints, scores):
        # 图像路径
        image_features = self.resnet(image)
        image_features = F.relu(self.image_fc(image_features))
        image_features = self.bn(image_features)
        image_features = self.dropout(image_features)

        # 应用通道注意力
        attention_weights = self.channel_attention(image_features.unsqueeze(2).unsqueeze(3))
        image_features = image_features * attention_weights.squeeze(-1).squeeze(-1)

        # 图像分类
        image_probabilities = F.softmax(self.image_classification_fc(image_features), dim=1)

        # 关键点路径
        keypoints = keypoints.view(keypoints.size(0), -1, 2)  # 调整关键点的维度
        scores = scores.view(scores.size(0), -1, 1)  # 调整分数的维度
        combined_keypoints = self.keypoint_attention(keypoints, scores)

        keypoint_features = F.relu(self.keypoint_fc(combined_keypoints))
        keypoint_features = self.bn(keypoint_features)
        keypoint_features = self.dropout(keypoint_features)

        # 关键点分类
        keypoint_probabilities = F.softmax(self.keypoint_classification_fc(keypoint_features), dim=1)

        # 融合策略：使用简单的平均
        combined_probabilities = (image_probabilities + keypoint_probabilities) / 2.0

        return combined_probabilities

