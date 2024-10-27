import os
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
from torch.utils.data import Dataset
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from src.util import plot_confusion_matrix, save_classification_report
from trys.DriverBehaviorClassifier import DriverBehaviorClassifierWithAttention
from tqdm import tqdm  # 导入 tqdm

# 配置环境变量
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

# 定义测试数据集
class DistractedDriverTestDataset(Dataset):
    def __init__(self, features_dir, image_dir, transform=None):
        self.features_dir = features_dir
        self.image_dir = image_dir
        self.transform = transform or transforms.Compose([
            transforms.ToTensor(),
        ])
        self.class_names = sorted([d for d in os.listdir(image_dir) if os.path.isdir(os.path.join(image_dir, d))])
        self.images_per_class = {}

        for class_name in self.class_names:
            class_images = os.listdir(os.path.join(image_dir, class_name))
            self.images_per_class[class_name] = class_images

    def __len__(self):
        return sum([len(self.images_per_class[c]) for c in self.class_names])

    def __getitem__(self, idx):
        class_idx = idx // len(self.images_per_class[self.class_names[0]])
        class_name = self.class_names[class_idx]
        img_idx = idx % len(self.images_per_class[class_name])
        img_name = self.images_per_class[class_name][img_idx]
        features_path = os.path.join(self.features_dir, class_name, img_name.replace('.jpg', '_features.npy'))

        # 加载关键点数据
        keypoints_data = np.load(features_path)

        # 检查关键点数据的形状
        if keypoints_data.ndim != 3 or keypoints_data.shape[1] != 133 or keypoints_data.shape[2] != 3:
            raise ValueError(f"Keypoints data must have shape (1, 133, 3), but got {keypoints_data.shape}")

        # 分离关键点坐标和置信度分数
        keypoints = torch.tensor(keypoints_data[0, :, :2], dtype=torch.float32)  # 坐标
        scores = torch.tensor(keypoints_data[0, :, 2], dtype=torch.float32)  # 置信度分数

        label = self.class_names.index(class_name)

        # 打开图像文件
        img_path = os.path.join(self.image_dir, class_name, img_name)
        image = Image.open(img_path).convert('RGB')

        # 应用转换
        image = self.transform(image)

        return image, keypoints, scores, torch.tensor(label, dtype=torch.long)

# 定义测试函数
def test_model(model, dataloader, device, class_names, save_path):
    model.eval()
    all_predictions = []
    all_true_labels = []

    with torch.no_grad():
        for images, keypoints, scores, labels in tqdm(dataloader, desc="Testing"):
            images = images.to(device)
            keypoints = keypoints.to(device)
            scores = scores.to(device)
            labels = labels.to(device)

            outputs = model(images, keypoints, scores)
            _, predicted = torch.max(outputs, 1)
            all_predictions.extend(predicted.cpu().numpy())
            all_true_labels.extend(labels.cpu().numpy())

    # 保存预测结果
    predictions_df = pd.DataFrame(all_predictions, columns=['predicted_class'])
    predictions_df['predicted_class_name'] = predictions_df['predicted_class'].map(
        {i: class_names[i] for i in range(len(class_names))})
    predictions_df.to_csv(os.path.join(save_path, 'predictions.csv'), index=False)

    # 如果有真实标签，则计算并保存性能指标
    if all_true_labels:
        true_labels = np.array(all_true_labels)
        plot_confusion_matrix1(true_labels, all_predictions, class_names,
                               os.path.join(save_path, 'confusion_matrix_percent.png'), normalize='true')
        plot_binary_confusion_matrix(true_labels, all_predictions, os.path.join(save_path, 'binary_confusion_matrix_percent.png'))
        save_classification_report1(true_labels, all_predictions, class_names, save_path)


def plot_confusion_matrix1(y_true, y_pred, class_names, save_path, normalize=None):
    cm = confusion_matrix(y_true, y_pred, labels=range(len(class_names)), normalize=normalize)
    cm_percentage = (cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]) * 100
    disp = ConfusionMatrixDisplay(confusion_matrix=cm_percentage, display_labels=class_names)
    fig, ax = plt.subplots(figsize=(10, 10))
    disp.plot(cmap=plt.cm.Blues, ax=ax, values_format='.2f' if normalize else 'd')
    plt.xticks(rotation=45, ha='right')  # 旋转 x 轴标签
    plt.tight_layout()  # 调整布局以防止标签重叠
    plt.savefig(save_path)
    plt.close(fig)


def plot_binary_confusion_matrix(y_true, y_pred, save_path):
    # 确保输入是 NumPy 数组
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # 将预测值和真实标签转换为二元标签
    binary_true_labels = (y_true == 1).astype(int)  # 1代表Distracted，0代表Safe Driving
    binary_y_pred = (y_pred == 1).astype(int)       # 1代表Distracted，0代表Safe Driving

    # 计算混淆矩阵
    cm = confusion_matrix(binary_true_labels, binary_y_pred, labels=[1, 0])

    # 创建一个新的 DataFrame 来存储带有百分比的混淆矩阵
    cm_with_percentages = pd.DataFrame(cm, index=['Safe Driving', 'Distracted Driving'], columns=['Safe Driving', 'Distracted Driving'])
    cm_with_percentages /= cm.sum().sum()

    # 绘制混淆矩阵
    fig, ax = plt.subplots(figsize=(8, 8))
    cax = ax.matshow(cm_with_percentages, cmap=plt.cm.Blues)
    fig.colorbar(cax)

    # 设置轴标签
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')

    # 设置标题
    ax.set_title('Binary Confusion Matrix with Percentages')

    # 设置 x 轴和 y 轴标签
    ax.set_xticks(np.arange(2))
    ax.set_yticks(np.arange(2))
    ax.set_xticklabels(['Safe Driving', 'Distracted Driving'])
    ax.set_yticklabels(['Safe Driving', 'Distracted Driving'])

    # 旋转 x 轴标签并将其放置在底部
    ax.xaxis.set_ticks_position('bottom')
    ax.xaxis.set_label_position('bottom')
    # 旋转 x 轴标签并将其放置在底部
    plt.xticks(np.arange(2), ['Safe Driving', 'Distracted Driving'], rotation=45)
    plt.yticks(np.arange(2), ['Safe Driving', 'Distracted Driving'])

    # 调整图形布局
    plt.tight_layout()

    # 添加百分比文本到每个单元格
    for i in range(2):
        for j in range(2):
            percentage = f'{cm_with_percentages.iloc[i, j] * 100:.2f}'
            plt.text(j, i, percentage,
                     horizontalalignment='center', verticalalignment='center',
                     fontsize=12)

    # 保存图像
    plt.savefig(save_path)
    plt.close(fig)

def save_classification_report1(y_true, y_pred, class_names, save_path):
    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    df = pd.DataFrame(report).transpose()
    df.index.name = 'class_name'  # 设置索引名为 class_name
    df.reset_index(inplace=True)  # 将索引转换为列
    df.to_csv(os.path.join(save_path, 'classification_report.csv'), index=False)


if __name__ == '__main__':
    # 配置
    features_test_dir = 'datasets/state-farm-distracted-driver-detection/test/features'
    images_test_dir = 'datasets/state-farm-distracted-driver-detection/imgs/test'
    model_weights_path = 'runs/train_all/best_model.pth'
    batch_size = 8
    num_workers = 4
    class_names = ['safe driving', 'texting - right', 'texting - left', 'talking on the phone - right',
                   'talking on the phone - left', 'operating the radio', 'drinking', 'reaching behind',
                   'hair and makeup', 'talking to passenger']
    save_path = 'runs/train_all/test_results/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # 创建数据加载器
    test_dataset = DistractedDriverTestDataset(features_test_dir, images_test_dir)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    # 模型和设备配置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DriverBehaviorClassifierWithAttention().to(device)
    model.load_state_dict(torch.load(model_weights_path, map_location=device))

    # 开始测试
    test_model(model, test_dataloader, device, class_names, save_path)