from PIL import Image
import tensorflow as tf
import numpy as np



def get_cover_images(i, batch_size):
    temp=[]
    for j in range(i,i+batch_size):
        with Image.open(f"datasets/qrcodes/c{j+1}.png").convert("L") as img:
            temp.append(img)
    temp = tf.convert_to_tensor(np.array(temp), dtype='float32')
    temp = tf.expand_dims(temp, axis=-1)
    # temp = tf.repeat(temp, repeats=3, axis=-1)
    # print()
    return temp

def get_cover_images3(i, batch_size):
    temp=[]
    for j in range(i,i+batch_size):
        with Image.open(f"datasets/qrcodes/c{j+1}.png").convert("L") as img:
            temp.append(img)
    temp = tf.convert_to_tensor(np.array(temp), dtype='float32')
    temp = tf.expand_dims(temp, axis=-1)
    temp = tf.repeat(temp, repeats=3, axis=-1)
    # print()
    return temp

def get_watermark_images(i, batch_size):
    temp=[]
    for j in range(i,i+batch_size):
        with Image.open(f"datasets/train/{j+1}.png").convert("L") as img:
            temp.append(img)

    temp = tf.convert_to_tensor(np.array(temp), dtype='float32')
    temp = tf.expand_dims(temp, axis=-1)
    # temp = tf.repeat(temp, repeats=3, axis=-1)
    return temp

# def get_watermark_images3(i, batch_size):
#     temp=[]
#     for j in range(i,i+batch_size):
#         with Image.open(f"datasets/train/{j+1}.png").convert("L") as img:
#             temp.append(img)
#
#     temp = tf.convert_to_tensor(np.array(temp), dtype='float32')
#     temp = tf.expand_dims(temp, axis=-1)
#     temp = tf.repeat(temp, repeats=3, axis=-1)
#     return temp


def get_watermark_images3(i, batch_size):
    temp=[]
    for j in range(i,i+batch_size):
        with Image.open(f"datasets/train/{j+1}.png").convert("L") as img:
            temp.append(img)
    temp = tf.convert_to_tensor(np.array(temp), dtype='float32')
    temp = tf.expand_dims(temp, axis=-1)
    temp = tf.repeat(temp, repeats=3, axis=-1)
    # print()
    return temp




if __name__ == '__main__':
    images = get_cover_images(1,4)
    from keras.utils import array_to_img
    print(array_to_img(images[0]).save('test.png'))