import numpy as np
import cv2
from keras.saving.save import load_model
from utils import *


def calculate_nc(img1, img2):
    """
     计算两张图像的归一化互相关(NC)值
     :param img1: 第一张图像（numpy数组）
     :param img2: 第二张图像（numpy数组）
     :return: 归一化互相关值（NC值）
     """
    # 确保图像为float类型，并归一化到[0, 1]范围
    img1 = img1.astype(np.float64) / 255.0
    img2 = img2.astype(np.float64) / 255.0

    # 将图像展平为一维数组
    img1_flat = img1.flatten()
    img2_flat = img2.flatten()

    # 计算归一化互相关值
    numerator = np.sum((img1_flat - np.mean(img1_flat)) * (img2_flat - np.mean(img2_flat)))
    denominator = np.sqrt(np.sum((img1_flat - np.mean(img1_flat)) ** 2) * np.sum((img2_flat - np.mean(img2_flat)) ** 2))

    if denominator == 0:
        return 0

    return numerator / denominator

def calculate_nc2(img1, img2):
    """
    计算两张图像的归一化互相关(NC)值
    :param img1: 第一张图像（numpy数组）
    :param img2: 第二张图像（numpy数组）
    :return: 归一化互相关值（NC值）
    """
    # 将图像展平为一维数组
    img1_flat = img1.flatten()
    img2_flat = img2.flatten()

    # 计算归一化互相关值
    numerator = np.sum((img1_flat - np.mean(img1_flat)) * (img2_flat - np.mean(img2_flat)))
    denominator = np.sqrt(np.sum((img1_flat - np.mean(img1_flat)) ** 2) * np.sum((img2_flat - np.mean(img2_flat)) ** 2))

    if denominator == 0:
        return 0

    return numerator / denominator

import tensorflow as tf
import time
if __name__ == '__main__':
    attact_list = ['gaussian', 'revolve', 'salt', 'shifted','jpeg']
    # decoder = make_decoder()
    # decoder.load_weights("../logs/desg_model_two/weights/decoder_epoch_20.h5")
    img1 = cv2.imread(r'..\logs\desg_model\test\decoded_s.png', cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(r'..\datasets\train\7.png', cv2.IMREAD_GRAYSCALE)
    print(img1.shape)
    print(img2.shape)
    nc_value = calculate_nc2(img1, img2)
    print(nc_value)
    # for attact in attact_list:
    #     print(f"{attact} NC")
    #     for i in range(5):
    #         if attact == 'jpeg':
    #             img2 = cv2.imread(f'../test/images/{attact}{i}.jpg', cv2.IMREAD_GRAYSCALE)  # 以灰度模式读取
    #         else:
    #             img2 = cv2.imread(f'../test/images/{attact}{i}.png', cv2.IMREAD_GRAYSCALE)  # 以灰度模式读取
    #         temp = tf.convert_to_tensor(np.array(img2), dtype='float32')
    #         temp = tf.expand_dims(temp, axis=-1)
    #         temp = tf.expand_dims(temp, axis=0)
    #         temp = tf.repeat(temp, repeats=3, axis=-1)
    #         img2 = decoder(temp)
    #         img2 = np.array(img2[0,:,:,0])
    #         nc_value = calculate_nc(img1, img2)
    #         print(f'{attact}{i}:',round(nc_value,4))
    #         time.sleep(0.5)






# import cv2
# import numpy as np
#
#
# # 计算归一化互相关系数的函数
# def calculate_nc(image1, image2):
#     # 转为浮点型
#     image1 = image1.astype(np.float32)
#     image2 = image2.astype(np.float32)
#
#     # 归一化
#     mean1 = np.mean(image1)
#     mean2 = np.mean(image2)
#     normalized1 = image1 - mean1
#     normalized2 = image2 - mean2
#
#     # 计算NC值
#     numerator = np.sum(normalized1 * normalized2)
#     denominator = np.sqrt(np.sum(normalized1 ** 2) * np.sum(normalized2 ** 2))
#
#     return numerator / denominator if denominator != 0 else 0
#
#
# # 旋转图像的函数
# def rotate_image(image, angle):
#     (h, w) = image.shape[:2]
#     center = (w // 2, h // 2)
#     matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
#     rotated = cv2.warpAffine(image, matrix, (w, h))
#     return rotated
#
#
# # 在一定角度范围内搜索最大NC值
# def find_best_rotation(image_a, image_b, angle_range=(-10, 10), step=0.5):
#     best_nc = -1
#     best_angle = 0
#
#     for angle in np.arange(angle_range[0], angle_range[1], step):
#         rotated_image_a = rotate_image(image_a, angle)
#         nc_value = calculate_nc(rotated_image_a, image_b)
#         print(f"最佳匹配角度: {angle}°")
#         print(f"最佳匹配的NC值: {nc_value}")
        # if nc_value > best_nc:
        #     best_nc = nc_value
        #     best_angle = angle

    # return best_angle, best_nc


# 加载图像
# image_a = cv2.imread('D:\digitalwatermark\deep-steg\watermarks\s3.png', cv2.IMREAD_GRAYSCALE)
# image_b = cv2.imread('xuanzhuan30.png', cv2.IMREAD_GRAYSCALE)

# 找到最佳匹配角度和对应的NC值
# best_angle, best_nc = find_best_rotation(image_a, image_b)

# print(f"最佳匹配角度: {best_angle}°")
# print(f"最佳匹配的NC值: {best_nc}")
