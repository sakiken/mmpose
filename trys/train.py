from mmpose.apis import inference_topdown, init_model
from mmpose.utils import register_all_modules
import cv2
import os
import pandas as pd
import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np
import torch.nn as nn

from trys.DistractedDriverDatasetWithKeypoints import DistractedDriverDatasetWithKeypoints
from trys.DriverBehaviorClassifier import DriverBehaviorClassifier

register_all_modules()
from PIL import Image
from torchvision.models import resnet50



if __name__ == '__main__':
    # 初始化人体和手部关键点检测模型
    config_file = 'projects/rtmpose/rtmpose/wholebody_2d_keypoint/rtmpose-l_8xb32-270e_coco-wholebody-384x288.py'
    checkpoint_file = 'weights/rtmpose/rtmpose-l_simcc-ucoco_dw-ucoco_270e-384x288-2438fd99_20230728.pth'

    image_dir = 'datasets/state-farm-distracted-driver-detection/imgs/train'
    csv_file = 'datasets/state-farm-distracted-driver-detection/driver_imgs_list.csv'
    # 创建数据集和数据加载器
    register_all_modules()

    dataset = DistractedDriverDatasetWithKeypoints(image_dir=image_dir ,csv_file=csv_file,config_file=config_file,checkpoint_file=checkpoint_file)
    print(f"Number of samples: {len(dataset)}")
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=4)
    print(f"Number of batches in dataloader: {len(dataloader)}")  # 添加这行
    # 初始化模型
    model = DriverBehaviorClassifier().to('cuda')
    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    num_epochs = 10
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for images, keypoints, keypoint_scores, labels in dataloader:
            images, keypoints, keypoint_scores, labels = images.to('cuda'), keypoints.to('cuda'), keypoint_scores.to(
                'cuda'), labels.to('cuda')

            optimizer.zero_grad()
            outputs = model(images, keypoints, keypoint_scores)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(dataloader):.4f}')

    print('Training complete.')
