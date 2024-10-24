from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from mmpose.utils import register_all_modules
import os
import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from tqdm import tqdm

from src.util import save_log_file, save_classification_report, plot_training_loss, plot_metrics, plot_confusion_matrix, \
    create_run_directory
from trys.DistractedDriverDatasetWithKeypoints import DistractedDriverDatasetWithKeypoints
from trys.DriverBehaviorClassifier import DriverBehaviorClassifier, DriverBehaviorClassifierWithCrossAttention

register_all_modules()
from PIL import Image
from torchvision.models import resnet50

def train_model(model, dataloader, criterion, optimizer, num_epochs, save_path=None, log_file=None):
    model.train()
    best_loss = float('inf')
    all_labels = []
    all_predictions = []
    epoch_losses = []  # 用于存储每个epoch的平均损失
    epoch_accuracies = []
    epoch_precisions = []
    epoch_recalls = []
    epoch_f1_scores = []

    with open(log_file, 'w') as f:
        for epoch in range(num_epochs):
            running_loss = 0.0
            all_labels = []
            all_predictions = []

            with tqdm(dataloader, unit="batch") as tepoch:
                for i, (images, keypoints, keypoint_scores, labels) in enumerate(tepoch):
                    labels = labels.long()
                    images = images.float()
                    keypoints = keypoints.float()
                    keypoint_scores = keypoint_scores.float()
                    if torch.cuda.is_available():
                        images = images.cuda()
                        keypoints = keypoints.cuda()
                        keypoint_scores = keypoint_scores.cuda()
                        labels = labels.cuda()

                    # 检查输入数据是否有 NaN 值
                    if torch.isnan(images).any() or torch.isnan(keypoints).any() or torch.isnan(
                            keypoint_scores).any() or torch.isnan(labels).any():
                        print(f"NaN values detected in input data at batch {i}")
                        continue

                    optimizer.zero_grad()
                    outputs = model(images, keypoints, keypoint_scores)
                    loss = criterion(outputs, labels)

                    # 检查 loss 是否为 NaN
                    if torch.isnan(loss).any():
                        print(f"Loss is NaN at batch {i}")
                        continue

                    loss.backward()
                    # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # 添加梯度裁剪
                    optimizer.step()
                    running_loss += loss.item()

                    _, predicted = torch.max(outputs, 1)
                    all_labels.extend(labels.cpu().numpy())
                    all_predictions.extend(predicted.cpu().numpy())

                    # 更新进度条描述信息
                    tepoch.set_description(f"Epoch [{epoch + 1}/{num_epochs}]")
                    tepoch.set_postfix(loss=loss.item())

                    # 打印每个批次的损失信息
                    # print(f"Epoch [{epoch + 1}/{num_epochs}], Batch {i + 1}/{len(dataloader)}, Loss: {loss.item():.4f}")
                    f.write(f"Epoch [{epoch + 1}/{num_epochs}], Batch {i + 1}/{len(dataloader)}, Loss: {loss.item():.4f}\n")

            epoch_loss = running_loss / len(dataloader)
            epoch_losses.append(epoch_loss)
            epoch_accuracy = accuracy_score(all_labels, all_predictions)
            epoch_precision = precision_score(all_labels, all_predictions, average='weighted')
            epoch_recall = recall_score(all_labels, all_predictions, average='weighted')
            epoch_f1 = f1_score(all_labels, all_predictions, average='weighted')

            epoch_accuracies.append(epoch_accuracy)
            epoch_precisions.append(epoch_precision)
            epoch_recalls.append(epoch_recall)
            epoch_f1_scores.append(epoch_f1)

            log_message = f"Epoch [{epoch + 1}/{num_epochs}], Average Loss: {epoch_loss:.6f}, Accuracy: {epoch_accuracy:.6f}, Precision: {epoch_precision:.6f}, Recall: {epoch_recall:.6f}, F1 Score: {epoch_f1:.6f}"
            print(log_message)
            f.write(log_message + '\n')

            # 保存每个epoch的信息到文件
            with open(os.path.join(save_path, 'epoch_info.txt'), 'a') as epoch_file:
                epoch_file.write(log_message + '\n')

            if epoch_loss < best_loss:
                best_loss = epoch_loss
                torch.save(model.state_dict(), os.path.join(save_path, 'model.pth'))

    # 保存日志文件
    save_log_file(epoch_losses, save_path)

    # 计算混淆矩阵
    class_names = [
        'safe driving', 'texting - right', 'texting - left', 'talking on the phone - right',
        'talking on the phone - left', 'operating the radio', 'drinking', 'reaching behind',
        'hair and makeup', 'talking to passenger'
    ]
    plot_confusion_matrix(all_labels, all_predictions, class_names, save_path)

    # 保存分类报告
    save_classification_report(all_labels, all_predictions, class_names, save_path)

    # 绘制损失曲线
    plot_training_loss(epoch_losses, save_path)

    # 绘制准确率、召回率和 F1 分数曲线
    plot_metrics(epoch_accuracies, epoch_precisions, epoch_recalls, epoch_f1_scores, save_path)


if __name__ == '__main__':
    # 初始化人体和手部关键点检测模型
    config_file = 'projects/rtmpose/rtmpose/wholebody_2d_keypoint/rtmpose-l_8xb32-270e_coco-wholebody-384x288.py'
    checkpoint_file = 'weights/rtmpose/rtmpose-l_simcc-ucoco_dw-ucoco_270e-384x288-2438fd99_20230728.pth'

    image_dir = 'datasets/state-farm-distracted-driver-detection/imgs/train'
    csv_file = 'datasets/state-farm-distracted-driver-detection/driver_imgs_list.csv'
    # 创建数据集和数据加载器
    dataset = DistractedDriverDatasetWithKeypoints(image_dir=image_dir ,csv_file=csv_file,config_file=config_file,checkpoint_file=checkpoint_file)
    print(f"Number of samples: {len(dataset)}")
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
    print(f"Number of batches in dataloader: {len(dataloader)}")  # 添加这行
    # 初始化模型

    run_dir = create_run_directory()
    log_file = os.path.join(run_dir, 'training_log.txt')
    # model = DriverBehaviorClassifier(num_classes=10).to('cuda')
    model = DriverBehaviorClassifierWithCrossAttention(num_classes=10).to('cuda')

    if torch.cuda.is_available():
        model = model.cuda()
        print("Using GPU")
    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    train_model(model, dataloader, criterion, optimizer, num_epochs=1, save_path=run_dir, log_file=log_file)
