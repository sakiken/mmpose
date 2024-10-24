from mmpose.apis import inference_topdown, init_model
from mmpose.utils import register_all_modules
import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np
register_all_modules()
from PIL import Image


class DistractedDriverDatasetWithKeypoints(Dataset):
    def __init__(self, image_dir, csv_file, config_file, checkpoint_file, num_samples=None):
        self.image_dir = image_dir
        self.data_frame = pd.read_csv(csv_file)
        self.config_file = config_file
        self.checkpoint_file = checkpoint_file
        print(1)
        self.transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        self.label_mapping = {'c0': 0, 'c1': 1, 'c2': 2, 'c3': 3, 'c4': 4,
                              'c5': 5, 'c6': 6, 'c7': 7, 'c8': 8, 'c9': 9}

        if num_samples is not None:
            self.data_frame = self.data_frame.head(num_samples)

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        img_name = self.data_frame.iloc[idx, 2]  # img
        class_name = self.data_frame.iloc[idx, 1]  # classname
        subject = self.data_frame.iloc[idx, 0]  # subject

        # 打开图像文件
        img_path = os.path.join(self.image_dir, class_name, img_name)
        image = Image.open(img_path).convert('RGB')

        # 将图像转换为 OpenCV 格式
        image_cv = np.array(image)
        model = init_model(self.config_file, self.checkpoint_file, device='cuda:0')
        # 使用 RTMPose 提取关键点
        register_all_modules()
        results = inference_topdown(model, image_cv)
        if results:
            result = results[0]
            pred_instances = result.pred_instances.to_dict()
            keypoints = pred_instances['keypoints'][0]
            keypoint_scores = pred_instances['keypoint_scores'][0]
        else:
            keypoints = np.zeros((133, 2))  # 如果没有检测到关键点，返回零向量
            keypoint_scores = np.zeros(133)

        # 将关键点转换为 PyTorch 张量
        keypoints = torch.tensor(keypoints, dtype=torch.float32)
        keypoint_scores = torch.tensor(keypoint_scores, dtype=torch.float32)

        # 应用转换
        image = self.transform(image)

        # 获取标签
        label = self.label_mapping[class_name]

        assert image.shape == (3, 224, 224), f"Image shape is incorrect: {image.shape}"
        assert keypoints.shape == (133, 2), f"Keypoints shape is incorrect: {keypoints.shape}"
        assert keypoint_scores.shape == (133,), f"Keypoint scores shape is incorrect: {keypoint_scores.shape}"
        assert isinstance(label, int), f"Label is not an integer: {label}"

        return image, keypoints, keypoint_scores, torch.tensor(label, dtype=torch.long)