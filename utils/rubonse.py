import cv2
import numpy as np
from PIL import Image
from skimage.util import random_noise
from scipy.ndimage import rotate, shift, zoom
import os


def gaussian_noise(img, mean, sigma):
    '''
    此函数用将产生的高斯噪声加到图片上
    传入:
        img   :  原图
        mean  :  均值
        sigma :  标准差
    返回:
        gaussian_out : 噪声处理后的图片
        noise        : 对应的噪声
    '''
    # 将图片灰度标准化
    img = img / 255
    # 产生高斯 noise
    noise = np.random.normal(mean, sigma, img.shape)
    # 将噪声和图片叠加
    gaussian_out = img + noise
    # 将超过 1 的置 1，低于 0 的置 0
    gaussian_out = np.clip(gaussian_out, 0, 1)
    # 将图片灰度范围的恢复为 0-255
    gaussian_out = np.uint8(gaussian_out*255)
    # 将噪声范围搞为 0-255
    # noise = np.uint8(noise*255)
    return gaussian_out # 这里也会返回噪声，注意返回值
#

def saltpepper_noise(image, proportion):
    '''
    此函数用于给图片添加椒盐噪声
    image       : 原始图片
    proportion  : 噪声比例
    '''
    image_copy = image.copy()
    # 求得其高宽
    img_Y, img_X, img_C = image.shape
    # 噪声点的 X 坐标
    X = np.random.randint(img_X, size=(int(proportion * img_X * img_Y),))
    # 噪声点的 Y 坐标
    Y = np.random.randint(img_Y, size=(int(proportion * img_X * img_Y),))
    # 噪声点的坐标赋值
    image_copy[Y, X,:] = np.random.choice([0, 255], size=(int(proportion * img_X * img_Y),3))

    # 噪声容器
    sp_noise_plate = np.ones_like(image_copy) * 127
    # 将噪声给噪声容器
    sp_noise_plate[Y, X,:] = image_copy[Y, X,:]
    return image_copy  # 这里也会返回噪声，注意返回值


def rotate_image(image1, angle):
    # 获取图片尺寸
    (h, w) = image1.shape[:2]
    # 图片中心
    center = (w // 2, h // 2)
    # 旋转矩阵
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    # 执行旋转
    rotated = cv2.warpAffine(image1, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return rotated


def shifted_img(image,size):
    # 定义平移矩阵
    # dx 和 dy 是平移的像素值，正值表示向右或向下平移，负值表示向左或向上平移
    dx = size[0]  # 水平方向平移的像素数
    dy = size[1]  # 垂直方向平移的像素数

    # 获取图像的尺寸
    rows, cols = image.shape[:2]

    # 创建平移矩阵
    M = np.float32([[1, 0, dx], [0, 1, dy]])

    # 应用平移矩阵
    shifted_image = cv2.warpAffine(image, M, (cols, rows))
    return shifted_image


def jpeg_compression_attack(image, output_path, quality=10):
    """
    对图像进行JPEG压缩攻击，模拟质量损失。

    :param image_path: 输入图像的路径
    :param output_path: 输出图像的路径
    :param quality: JPEG压缩质量，范围从0到100，值越小质量越差
    """
    # 读取图像
    image = Image.open(image_path)

    # 将图像保存为JPEG格式，并调整质量参数
    image.save(output_path, 'JPEG', quality=quality)
    # print(f"图像已保存为压缩版本，质量={quality}，路径：{output_path}")

# 测试
if __name__ == "__main__":
    attack_list = ['gaussian', 'salt_pepper', 'rotate', 'shift', 'zoom']
    d = "../test/images/"
    for i in range(1):
        # 加载原始图像
        image_path = r"..\logs\desg_model_two\test\decoded_cpred10400.png"
        img = cv2.imread(image_path)  # 替换为你的水印图像路径

        vars = [0.001,0.003,0.005,0.01,0.03,0.05]
        for num in range(len(vars)):
            # var = 0.01
            var = vars[num]
            gaussian_out1 = gaussian_noise(img, 0, var)
            gaussian_out1 = Image.fromarray(gaussian_out1)
            gaussian_out1.save(d+f'gaussian{num}.png')

        vars = [0.001, 0.004, 0.007, 0.01, 0.03, 0.05]
        for num in range(len(vars)):
            # var =  0.02
            var = vars[num]
            sp_out1 = saltpepper_noise(img, var)
            sp_out1 = Image.fromarray(sp_out1)
            sp_out1.save(d+f'salt{num}.png')

        vars = [1,2,3,4,5,6]
        for num in range(len(vars)):
            # var = 30
            var = vars[num]
            rotated_image = rotate_image(img, var)
            rotated_image = Image.fromarray(rotated_image)
            rotated_image.save(d+f'revolve{num}.png')

        vars = [[1,1],[1,2],[1,3], [2,1], [2,2], [3,3]]
        for num in range(len(vars)):
            # var = [1,2]
            var = vars[num]
            shifted_image = shifted_img(img, var)
            shifted_image = Image.fromarray(shifted_image)
            shifted_image.save(d+f'shifted{num}.png')

        vars = [90,80,70,60,50,40]
        for num in range(len(vars)):
            # var = [1,2]
            var = vars[num]
            jpeged_image = jpeg_compression_attack(image_path,d + f'jpeg{num}.jpg', var)





