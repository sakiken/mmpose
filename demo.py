from mmpose.apis import inference_topdown, init_model
from mmpose.utils import register_all_modules
import cv2
import numpy as np

register_all_modules()

# 初始化人体和手部关键点检测模型
config_file = 'projects/rtmpose/rtmpose/wholebody_2d_keypoint/rtmpose-l_8xb32-270e_coco-wholebody-384x288.py'
checkpoint_file = 'weights/rtmpose/rtmpose-l_simcc-ucoco_dw-ucoco_270e-384x288-2438fd99_20230728.pth'
model = init_model(config_file, checkpoint_file, device='cuda:0')


# 加载图片
image = cv2.imread('datasets/state-farm-distracted-driver-detection/imgs/train/c0/img_104.jpg')

# 使用模型进行推理
results = inference_topdown(model, image)
print(f"Results: {results}")
# 获取关键点
if results:
    # 获取第一个检测结果
    result = results[0]

    # 将 PoseDataSample 转换为字典
    pred_instances = result.pred_instances.to_dict()
    # print(pred_instances)
    keypoints = pred_instances['keypoints'][0]  # 去掉最外层的维度
    keypoint_scores = pred_instances['keypoint_scores'][0]

    # 打印所有关键点坐标
    # print("All Keypoints:", keypoints)

    # 绘制关键点
    for i, (x, y) in enumerate(keypoints):
        if keypoint_scores[i] > 0.3:  # 只绘制置信度较高的关键点
            cv2.circle(image, (int(x), int(y)), 2, (204, 255, 255), -1)

    # 显示带有关键点标注的图像
    cv2.imshow('Keypoints', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("No human detected in the image.")