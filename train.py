from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn as nn
from tqdm import tqdm
from mmpose.apis import inference_topdown, init_model
from mmpose.utils import register_all_modules
from src.util import save_log_file, save_classification_report, plot_training_loss, plot_metrics, plot_confusion_matrix, \
    create_run_directory
from trys.DistractedDriverDatasetWithKeypoints import DistractedDriverDatasetWithKeypoints
from trys.DriverBehaviorClassifier import DriverBehaviorClassifierWithAttention

from PIL import Image
from torchvision.models import resnet50

def evaluate_model(model, dataloader, criterion):
    model.eval()
    running_loss = 0.0
    all_labels = []
    all_predictions = []

    with torch.no_grad(), tqdm(dataloader, desc="Evaluating", leave=False) as pbar:
        for images, keypoints, keypoint_scores, labels in pbar:
            labels = labels.long()
            images = images.float()
            keypoints = keypoints.float()
            keypoint_scores = keypoint_scores.float()
            if torch.cuda.is_available():
                images = images.cuda()
                keypoints = keypoints.cuda()
                keypoint_scores = keypoint_scores.cuda()
                labels = labels.cuda()

            outputs = model(images, keypoints, keypoint_scores)
            loss = criterion(outputs, labels)
            running_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

            pbar.set_postfix({'Loss': f'{loss.item():.4f}'})

    avg_loss = running_loss / len(dataloader)
    accuracy = accuracy_score(all_labels, all_predictions)
    precision = precision_score(all_labels, all_predictions, average='weighted')
    recall = recall_score(all_labels, all_predictions, average='weighted')
    f1 = f1_score(all_labels, all_predictions, average='weighted')

    return avg_loss, accuracy, precision, recall, f1, all_labels, all_predictions

def train_model(model, train_dataloader, val_dataloader, criterion, optimizer, num_epochs, save_path=None, log_file=None):
    best_val_loss = float('inf')
    epoch_losses = []
    val_losses = []
    val_accuracies = []
    val_precisions = []
    val_recalls = []
    val_f1_scores = []

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    with open(log_file, 'w') as f:
        for epoch in range(num_epochs):
            # Training phase
            model.train()
            running_train_loss = 0.0

            with tqdm(train_dataloader, unit="batch") as tepoch:
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
                    optimizer.step()
                    running_train_loss += loss.item()

                    tepoch.set_description(f"Epoch [{epoch + 1}/{num_epochs}]")
                    tepoch.set_postfix(loss=loss.item())

            epoch_loss = running_train_loss / len(train_dataloader)
            epoch_losses.append(epoch_loss)

            # Validation phase
            val_epoch_loss, val_accuracy, val_precision, val_recall, val_f1, all_val_labels, all_val_predictions = evaluate_model(model, val_dataloader, criterion)

            # Logging
            log_message = (
                f"Epoch [{epoch + 1}/{num_epochs}], "
                f"Train Loss: {epoch_loss:.6f}, "
                f"Val Loss: {val_epoch_loss:.6f}, Val Acc: {val_accuracy:.6f}, "
                f"Val Precision: {val_precision:.6f}, Val Recall: {val_recall:.6f}, Val F1: {val_f1:.6f}"
            )
            print(log_message)
            f.write(log_message + '\n')

            # Save the best model based on val loss
            if val_epoch_loss < best_val_loss:
                best_val_loss = val_epoch_loss
                torch.save(model.state_dict(), os.path.join(save_path, 'best_model.pth'))

            # Store metrics for plotting
            val_losses.append(val_epoch_loss)
            val_accuracies.append(val_accuracy)
            val_precisions.append(val_precision)
            val_recalls.append(val_recall)
            val_f1_scores.append(val_f1)

    # 保存日志文件
    save_log_file(epoch_losses, save_path)

    # 计算混淆矩阵
    class_names = [
        'safe driving', 'texting - right', 'texting - left', 'talking on the phone - right',
        'talking on the phone - left', 'operating the radio', 'drinking', 'reaching behind',
        'hair and makeup', 'talking to passenger'
    ]
    plot_confusion_matrix(all_val_labels, all_val_predictions, class_names, save_path)

    # 保存分类报告
    save_classification_report(all_val_labels, all_val_predictions, class_names, save_path)

    # 绘制损失曲线
    plot_training_loss(epoch_losses, save_path)

    # 绘制准确率、召回率和 F1 分数曲线
    plot_metrics(val_accuracies, val_precisions, val_recalls, val_f1_scores, save_path)


if __name__ == '__main__':
    # 图像位置
    train_image_dir = 'datasets/state-farm-distracted-driver-detection/imgs/train'
    val_image_dir = 'datasets/state-farm-distracted-driver-detection/imgs/val'
    test_image_dir = 'datasets/state-farm-distracted-driver-detection/imgs/test'

    # CSV 文件
    train_csv_file = 'datasets/state-farm-distracted-driver-detection/train/train_driver_imgs_list.csv'
    val_csv_file = 'datasets/state-farm-distracted-driver-detection/val/val_driver_imgs_list.csv'

    # 初始化人体和手部关键点检测模型
    config_file = 'projects/rtmpose/rtmpose/wholebody_2d_keypoint/rtmpose-l_8xb32-270e_coco-wholebody-384x288.py'
    checkpoint_file = 'weights/rtmpose/rtmpose-l_simcc-ucoco_dw-ucoco_270e-384x288-2438fd99_20230728.pth'

    # 初始化关键点模型
    register_all_modules()
    keypoint_model = init_model(config_file, checkpoint_file, device='cuda:0')

    # 创建数据集和数据加载器
    train_dataset = DistractedDriverDatasetWithKeypoints(image_dir=train_image_dir, csv_file=train_csv_file, keypoint_model=keypoint_model)
    val_dataset = DistractedDriverDatasetWithKeypoints(image_dir=val_image_dir, csv_file=val_csv_file, keypoint_model=keypoint_model)

    print(f"Number of training samples: {len(train_dataset)}")
    print(f"Number of validation samples: {len(val_dataset)}")

    train_dataloader = DataLoader(train_dataset, batch_size=2, shuffle=True,drop_last=True)
    val_dataloader = DataLoader(val_dataset, batch_size=2, shuffle=True,drop_last=True)

    run_dir = create_run_directory()
    log_file = os.path.join(run_dir, 'training_log.txt')
    model = DriverBehaviorClassifierWithAttention(num_classes=10).to('cuda')

    if torch.cuda.is_available():
        model = model.cuda()
        print("Using GPU")

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    train_model(model, train_dataloader, val_dataloader, criterion, optimizer, num_epochs=1, save_path=run_dir, log_file=log_file)