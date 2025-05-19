import math
import numpy as np
from PIL import Image
import tensorflow as tf

def get_image(path):
    temp = Image.open(path).convert("L")
    temp = tf.convert_to_tensor(np.array(temp), dtype='float32')
    temp = tf.expand_dims(temp, axis=-1)
    temp = tf.expand_dims(temp, axis=0)
    temp = tf.repeat(temp, repeats=3, axis=-1)
    return temp
def psnr_(img1, img2):
    mse = np.mean((img1 / 255.0 - img2 / 255.0) ** 2)
    if mse < 1e-10:
        return 100
    psnr2 = 20 * math.log10(1 / math.sqrt(mse))
    return psnr2



def calculate_ssim(img1, img2):
    """
    计算两幅图像的 SSIM（结构相似性指数）
    :param img1: 第一幅图像（numpy 数组）
    :param img2: 第二幅图像（numpy 数组）
    :return: SSIM 值（float）
    """
    if img1.shape != img2.shape:
        raise ValueError("两幅图像的尺寸必须一致")

    # 定义常数
    K1, K2 = 0.01, 0.03
    L = 255  # 假设图像的像素值范围是 0-255
    C1 = (K1 * L) ** 2
    C2 = (K2 * L) ** 2

    # 计算均值
    mu1 = np.mean(img1)
    mu2 = np.mean(img2)

    # 计算方差和协方差
    sigma1_sq = np.var(img1)
    sigma2_sq = np.var(img2)
    sigma12 = np.mean((img1 - mu1) * (img2 - mu2))

    # 计算 SSIM
    numerator = (2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)
    denominator = (mu1 ** 2 + mu2 ** 2 + C1) * (sigma1_sq + sigma2_sq + C2)
    ssim_value = numerator / denominator

    return ssim_value



if __name__ == '__main__':
    cover_ = get_image(f"../datasets/qrcodes/c3.png")
    secret_ = get_image(f"../logs/desg_model/test/decoded_c3.png")
    secret_3 = get_image(f"../logs/desg_model/test/decoded_c10.png")

    secret_2 = get_image(f"D:\digitalwatermark\deep-steg\decoded_c.png")

    psnr = psnr_(cover_, secret_)
    psnr2 = psnr_(cover_, secret_2)
    print("psnr1:",psnr)
    print("psnr2:",psnr2)

    ssim1= calculate_ssim(cover_, secret_)
    ssim2= calculate_ssim(cover_, secret_2)
    ssim3= calculate_ssim(cover_, secret_3)
    print("ssim1:",ssim1)
    print("ssim2:",ssim3)