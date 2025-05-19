import cv2
import numpy as np
from keras.applications import VGG16
from keras.utils import img_to_array
from keras.applications.vgg16 import preprocess_input
from sklearn.metrics.pairwise import cosine_similarity


def sift_feature_matching(img1, img2):
    sift = cv2.SIFT_create()

    # 将图像转换为灰度图
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # 检测关键点和描述符
    kp1, des1 = sift.detectAndCompute(gray1, None)
    kp2, des2 = sift.detectAndCompute(gray2, None)

    # 使用 FLANN 匹配器进行匹配
    index_params = dict(algorithm=1, trees=10)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)

    # 使用 Lowe's ratio test 来筛选好的匹配点
    good_matches = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good_matches.append(m)

    return len(good_matches)


# 使用 VGG16 提取图像的高层特征
def cnn_feature_extraction(img):
    model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    img_resized = cv2.resize(img, (224, 224))
    img_array = img_to_array(img_resized)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    features = model.predict(img_array)
    features = features.flatten()
    return features


# 根据图像尺寸估算 max_sift_matches
def estimate_max_sift_matches(img):
    # 假设每张图像最大匹配点数为 100 或 200
    max_matches = 100  # 设定一个最大匹配点数
    return max_matches


def calculate_image_similarity(image1_path, image2_path):
    # 读取图像
    img1 = cv2.imread(image1_path)
    img2 = cv2.imread(image2_path)

    # 检查图像是否成功加载
    if img1 is None:
        print(f"Error: Failed to load image1 from path: {image1_path}")
        return None
    if img2 is None:
        print(f"Error: Failed to load image2 from path: {image2_path}")
        return None

    # Step 1: 使用 SIFT 进行特征匹配
    sift_matches = sift_feature_matching(img1, img2)

    # Step 2: 使用 CNN 提取特征并计算余弦相似度
    cnn_features_1 = cnn_feature_extraction(img1)
    cnn_features_2 = cnn_feature_extraction(img2)

    cos_sim = cosine_similarity([cnn_features_1], [cnn_features_2])[0][0]

    # Step 3: 估算 max_sift_matches
    max_sift_matches = estimate_max_sift_matches(img1)

    # 归一化 SIFT 特征匹配数量
    normalized_sift_matches = sift_matches / max_sift_matches

    # 综合 SIFT 特征匹配和 CNN 相似度
    final_similarity = (normalized_sift_matches * 0) + (cos_sim * 1)

    return final_similarity


# 调用示例
image1_path = '../logs/desg_model/test/decoded_s.png'  # 替换为你的图片路径
# image2_path = '../datasets/train/7.png'  # 替换为你的图片路径
image2_path = '../logs/desg_model/test/decoded_shifted[1, 2].png'  # 替换为你的图片路径

similarity_score = calculate_image_similarity(image1_path, image2_path)
if similarity_score is not None:
    print(f"图像相似度: {similarity_score:.4f}")