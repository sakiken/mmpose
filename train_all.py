import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import os, sys, time
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tqdm import tqdm
from src.util import check_labels, create_run_directory, save_classification_report, save_log_file, \
    plot_confusion_matrix, plot_training_loss, plot_metrics
from torch.utils.data import DataLoader
from trys.DistractedDriverDatasetWithKeypoints import DistractedDriverDatasetWithKeypoints, DistractedDriverDataset1
from trys.DriverBehaviorClassifier import DriverBehaviorClassifierWithAttention

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

def evaluate_model(model, dataloader, criterion):
    model.eval()
    running_loss = 0.0
    all_labels = []
    all_predictions = []

    # 使用tqdm包装dataloader以显示进度条
    with torch.no_grad(), tqdm(dataloader, desc="Evaluating", leave=False) as pbar:
        for images, keypoints, labels in pbar:
            labels = labels.long()
            images = images.float()
            keypoints = keypoints.float()
            if torch.cuda.is_available():
                images = images.cuda()
                keypoints = keypoints.cuda()
                labels = labels.cuda()

            outputs = model(images, keypoints)
            loss = criterion(outputs, labels)
            running_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

            # 更新进度条描述
            pbar.set_postfix({'Loss': f'{loss.item():.4f}'})

    avg_loss = running_loss / len(dataloader)
    accuracy = accuracy_score(all_labels, all_predictions)
    precision = precision_score(all_labels, all_predictions, average='weighted')
    recall = recall_score(all_labels, all_predictions, average='weighted')
    f1 = f1_score(all_labels, all_predictions, average='weighted')

    return avg_loss, accuracy, precision, recall, f1, all_labels, all_predictions

def train_model(model, train_dataloader, criterion, optimizer, num_epochs, save_path=None, log_file=None):
    best_val_loss = float('inf')
    val_losses = []
    val_accuracies = []
    val_precisions = []
    val_recalls = []
    val_f1_scores = []
    all_labels = []
    all_predictions = []
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    with open(log_file, 'w') as f:
        for epoch in range(num_epochs):
            # Training phase
            model.train()
            running_train_loss = 0.0

            pbar = tqdm(train_dataloader, desc=f'Epoch [{epoch + 1}/{num_epochs}]', leave=False)

            for i, (images, keypoints,scores, labels) in enumerate(pbar):
                labels = labels.long()
                images = images.float()
                keypoints = keypoints.float()
                scores = scores.float()
                if torch.cuda.is_available():
                    images = images.cuda()
                    keypoints = keypoints.cuda()
                    scores = scores.cuda()
                    labels = labels.cuda()

                optimizer.zero_grad()
                outputs = model(images, keypoints,scores)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_train_loss += loss.item()

                pbar.set_postfix_str(f'Batch {i}, Loss: {loss.item():.4f}')
                _, predicted = torch.max(outputs, 1)
                all_labels.extend(labels.cpu().numpy())
                all_predictions.extend(predicted.cpu().numpy())

            avg_loss = running_train_loss / len(train_dataloader)
            accuracy = accuracy_score(all_labels, all_predictions)
            precision = precision_score(all_labels, all_predictions, average='weighted')
            recall = recall_score(all_labels, all_predictions, average='weighted')
            f1 = f1_score(all_labels, all_predictions, average='weighted')

            val_epoch_loss, val_accuracy, val_precision, val_recall, val_f1, all_val_labels, all_val_predictions = avg_loss, accuracy, precision, recall, f1, all_labels, all_predictions
           # val phase
           #  val_epoch_loss, val_accuracy, val_precision, val_recall, val_f1, all_val_labels, all_val_predictions = evaluate_model(model, train_dataloader, criterion)

            # Logging
            log_message = (
                f"Epoch [{epoch + 1}/{num_epochs}], "
                f"Val Loss: {val_epoch_loss:.6f}, Val Acc: {val_accuracy:.6f}"
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
    save_log_file(val_losses, save_path)

    # 保存分类报告
    class_names = [
        'safe driving', 'texting - right', 'texting - left', 'talking on the phone - right',
        'talking on the phone - left', 'operating the radio', 'drinking', 'reaching behind',
        'hair and makeup', 'talking to passenger'
    ]
    save_classification_report(all_val_labels, all_val_predictions, class_names, save_path)

    plot_confusion_matrix(all_val_labels, all_val_predictions, class_names, save_path)

    # 保存分类报告
    save_classification_report(all_val_labels, all_val_predictions, class_names, save_path)

    # 绘制损失曲线
    plot_training_loss(val_losses, save_path)

    # 绘制准确率、召回率和 F1 分数曲线
    plot_metrics(val_accuracies, val_precisions, val_recalls, val_f1_scores, save_path)


if __name__ == "__main__":
    # 图像位置
    train_image_dir = 'datasets/state-farm-distracted-driver-detection/imgs/trains'
    # val_image_dir = 'datasets/state-farm-distracted-driver-detection/imgs/val'
    test_image_dir = 'datasets/state-farm-distracted-driver-detection/imgs/test'

    # 特征位置
    train_features_dir = 'datasets/state-farm-distracted-driver-detection/trains/features/'
    # val_features_dir = 'datasets/state-farm-distracted-driver-detection/val/features/'

    # CSV 文件
    # train_csv_file = 'datasets/state-farm-distracted-driver-detection/train/train_driver_imgs_list.csv'
    train_csv_file = 'datasets/state-farm-distracted-driver-detection/driver_imgs_list.csv'
    val_csv_file = 'datasets/state-farm-distracted-driver-detection/val/val_driver_imgs_list.csv'

    # 创建数据集
    train_dataset = DistractedDriverDataset1(features_dir=train_features_dir, image_dir=train_image_dir, csv_file=train_csv_file)
    # val_dataset = DistractedDriverDataset1(features_dir=val_features_dir, image_dir=val_image_dir, csv_file=val_csv_file)

    # 创建数据加载器
    train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True, drop_last=True,num_workers=4)
    # val_dataloader = DataLoader(val_dataset, batch_size=8, shuffle=False, drop_last=True,num_workers=4)

    print(f"Number of training samples: {len(train_dataset)}")
    # print(f"Number of validation samples: {len(val_dataset)}")

    run_dir = create_run_directory()
    log_file = os.path.join(run_dir, 'training_log.txt')
    model = DriverBehaviorClassifierWithAttention().cuda()

    if torch.cuda.is_available():
        model = model.cuda()
        print("Using GPU")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    train_model(model, train_dataloader, criterion, optimizer, num_epochs=1, save_path=run_dir, log_file=log_file)
    # train_model(model, train_dataloader, val_dataloader, criterion, optimizer, num_epochs=2, save_path=run_dir, log_file=log_file)