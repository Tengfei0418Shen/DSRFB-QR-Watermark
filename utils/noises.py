import cv2
import tensorflow as tf
import numpy as np
import tensorflow_addons as tfa

def scalar_output_shape(input_shape):
    return input_shape

def dropout_blocks(x):
    noise = tf.random.uniform(shape=[1, 4, 4, 1], maxval=1, dtype=tf.float32, seed=None)
    noise = noise > 0.25
    noise = tf.cast(noise, tf.float32)
    return x * noise
def multiply_scalar(x, scalar):
    return x * tf.convert_to_tensor(scalar, tf.float32)

def calculate_psnr(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    return 20 * np.log10(PIXEL_MAX / np.sqrt(mse))


def salt_pepper_noise(x, salt_ratio=0.04):
    """
    添加 Salt-and-Pepper 噪声，保持输入和输出形状一致。
    Args:
        x: 输入张量，形状为 (batch_size, height, width, channels)。
        salt_ratio: 盐噪声的比例（0~1）。
    Returns:
        添加噪声后的张量，形状与输入一致。
    """
    # 获取输入形状
    shape = tf.shape(x)

    # 随机生成噪声掩码
    random_noise = tf.random.uniform(shape)

    # 创建盐（白色）和胡椒（黑色）掩码
    salt_mask = tf.cast(random_noise < salt_ratio / 2, dtype=x.dtype)  # 盐掩码
    pepper_mask = tf.cast(random_noise > 1 - salt_ratio / 2, dtype=x.dtype)  # 胡椒掩码

    # 添加噪声：盐变为 1，胡椒变为 0，其余保持原值
    x_salt_pepper = x * (1 - salt_mask - pepper_mask) + salt_mask
    return x_salt_pepper

def rotate_image(x):
    """ Perform random rotations (90, 180, 270 degrees) on input images """
    angles = [0, 90, 180, 270]
    angle = tf.random.shuffle(angles)[0]  # Randomly choose one angle
    return tf.image.rot90(x, k=angle // 90)  # k is 0, 1, 2, or 3

def rotate_image11(x):
    # Generate a random angle between -5 and 25 degrees
    angle = tf.random.uniform([], minval=-5, maxval=25, dtype=tf.float32)

    # Convert the angle to radians (as tf.image.rotate expects radians)
    angle_rad = angle * (3.141592653589793 / 180.0)

    # Apply the rotation
    rotated_image = tfa.image.rotate(x, angle_rad)

    return rotated_image


def random_switch(x, prob):
    # 确保输入的张量形状一致
    # target_shape = tf.shape(x[0])  # 假设所有张量都需要匹配 x[0] 的形状

    # salt_pepper_attacked, noise_attacked, mean_attacked
    # x0 = tf.image.resize(x[0], (target_shape[1], target_shape[2]))  # 调整空间大小
    # x1 = tf.image.resize(x[1], (target_shape[1], target_shape[2]))
    # x2 = tf.image.resize(x[2], (target_shape[1], target_shape[2]))
    # x3 = tf.image.resize(x[3], (target_shape[1], target_shape[2]))
    # x4 = tf.image.resize(x[4], (target_shape[1], target_shape[2]))

    num = 5
    noise_idx = np.random.randint(num)
    # print("~~~~~",noise_idx)
    return x[noise_idx]
    # noise0 = tf.cast(noise_idx%num, tf.float32)
    # noise1 = tf.cast(noise_idx%num, tf.float32)
    # noise2 = tf.cast(noise_idx%num, tf.float32)
    # noise3 = tf.cast(noise_idx%num, tf.float32)
    # noise4 = tf.cast(noise_idx%num, tf.float32)
    # print()
    # return x0 * noise0 + x1 * noise1 + x2 * noise2 + x3 * noise3 + x4 * noise4