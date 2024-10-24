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


class CrossAttention(nn.Module):
    def __init__(self, query_dim, key_dim, value_dim, embed_dim, num_heads=8):
        super(CrossAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.query = nn.Linear(query_dim, embed_dim)
        self.key = nn.Linear(key_dim, embed_dim)
        self.value = nn.Linear(value_dim, embed_dim)
        self.out = nn.Linear(embed_dim, embed_dim)

    def forward(self, query, key, value):
        B, Nq, _ = query.shape
        B, Nk, _ = key.shape
        q = self.query(query).view(B, Nq, self.num_heads, self.embed_dim // self.num_heads).permute(0, 2, 1, 3)
        k = self.key(key).view(B, Nk, self.num_heads, self.embed_dim // self.num_heads).permute(0, 2, 1, 3)
        v = self.value(value).view(B, Nk, self.num_heads, self.embed_dim // self.num_heads).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) / (self.embed_dim // self.num_heads) ** 0.5
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B, Nq, self.embed_dim)
        x = self.out(x)
        return x


class DriverBehaviorClassifierWithCrossAttention(nn.Module):
    def __init__(self, num_classes=10, num_keypoints=133):
        super(DriverBehaviorClassifierWithCrossAttention, self).__init__()

        # 加载ResNet-50模型，不使用预训练权重
        self.resnet = resnet50(weights=None)

        # 替换最后一层全连接层
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Identity()  # 移除原ResNet的最后一层

        # 全连接层
        self.image_fc = nn.Linear(num_features, 512)
        self.keypoint_fc = nn.Linear(num_keypoints * 2 + num_keypoints, 512)  # 关键点位置和分数的总维度

        # 交叉注意力机制
        self.cross_attention = CrossAttention(query_dim=512, key_dim=512, value_dim=512, embed_dim=512, num_heads=8)

        # 分类任务的全连接层
        self.classification_fc = nn.Linear(512, num_classes)

        # 添加 BatchNorm 和 Dropout
        self.bn = nn.BatchNorm1d(512)
        self.dropout = nn.Dropout(0.5)

    def forward(self, image, keypoints, keypoint_scores):
        # 通过ResNet主干网络处理图像
        image_features = self.resnet(image)

        # 处理图像特征
        image_features = F.relu(self.image_fc(image_features))
        image_features = self.bn(image_features)
        image_features = self.dropout(image_features)

        # 处理关键点特征
        keypoints = keypoints.view(keypoints.size(0), -1)
        keypoint_scores = keypoint_scores.view(keypoint_scores.size(0), -1)
        keypoint_features = torch.cat((keypoints, keypoint_scores), dim=1)
        keypoint_features = F.relu(self.keypoint_fc(keypoint_features))
        keypoint_features = self.bn(keypoint_features)
        keypoint_features = self.dropout(keypoint_features)

        # 应用交叉注意力机制
        attention_features = self.cross_attention(
            image_features.unsqueeze(1),
            keypoint_features.unsqueeze(1),
            keypoint_features.unsqueeze(1)
        ).squeeze(1)

        # 分类任务的输出
        classification_output = self.classification_fc(attention_features)

        return classification_output