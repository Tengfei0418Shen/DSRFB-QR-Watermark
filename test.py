from keras.utils import array_to_img
import matplotlib.pyplot as plt
from utils import *

beta = 0.2


def rev_loss(s_true, s_pred):
    extraction_loss = K.sum(K.square(s_true - s_pred))
    return 1.0 * extraction_loss


# # Loss for the full model (self-supervised)
def full_loss(y_true, y_pred):
    # Loss for the full model is: |C-C'| + beta * |S-S'|
    s_true, c_true = y_true[0], y_true[1]
    c_pred = y_pred

    s_loss = beta * K.sum(K.square(s_true - c_pred))
    c_loss = K.sum(K.square(c_true - c_pred))

    return s_loss + c_loss


def calculate_psnr(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    return 20 * np.log10(PIXEL_MAX / np.sqrt(mse))





def get_image(path):
    temp = Image.open(path).convert("L")
    temp = tf.convert_to_tensor(np.array(temp), dtype='float32')
    temp = tf.expand_dims(temp, axis=-1)
    temp = tf.expand_dims(temp, axis=0)
    temp = tf.repeat(temp, repeats=3, axis=-1)
    return temp

def get_image2(path):
    temp = Image.open(path).convert("L")
    temp = tf.convert_to_tensor(np.array(temp.resize((64, 64))), dtype='float32')
    temp = tf.expand_dims(temp, axis=-1)
    temp = tf.expand_dims(temp, axis=0)
    temp = tf.repeat(temp, repeats=3, axis=-1)
    return temp


def make_model(input_size):
    input_S = Input(shape=(input_size))
    input_C = Input(shape=(input_size))

    encoder = make_encoder(input_size)

    decoder = make_decoder(input_size)
    decoder.compile(optimizer=tf.keras.optimizers.Adam(0.001), loss=rev_loss)
    decoder.trainable = False

    output_Cprime = encoder([input_S, input_C])
    # print(output_Cprime)
    output_Sprime = decoder(output_Cprime)

    autoencoder = Model(
        inputs=[input_S, input_C], outputs=concatenate([output_Sprime, output_Cprime])
    )
    autoencoder.compile(optimizer=tf.keras.optimizers.Adam(0.001), loss=full_loss)

    return encoder, decoder, autoencoder


def calculate_psnr(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    return 30 * np.log10(PIXEL_MAX / np.sqrt(mse))

def calculate_psnr1(img1, img2):
    """
    计算两幅图像的PSNR值。
    :param img1: 第一幅图像，numpy数组格式。
    :param img2: 第二幅图像，numpy数组格式。
    :return: PSNR值。
    """
    # 确保两幅图像大小和类型相同
    if img1.shape != img2.shape:
        raise ValueError("两幅图像的大小和类型必须一致！")

    # 计算均方误差（MSE）
    mse = np.mean((img1 - img2) ** 2)

    if mse == 0:
        # 如果MSE为0，说明两幅图像完全相同，PSNR为无穷大
        return float('inf')

    # 最大可能的像素值（对于8位图像为255）
    max_pixel_value = 255.0

    # 计算PSNR
    psnr = 30 * np.log10(max_pixel_value / np.sqrt(mse))
    return psnr

if __name__ == '__main__':
    input_size = (64, 64, 3)  # Example input size

    d = "logs/desg_model_two/"

    encoder = make_encoder(input_size)
    encoder.load_weights(d + "weights/encoder_epoch_19.h5")
    decoder = make_decoder(input_size)
    decoder.load_weights(d + "weights/decoder_epoch_19.h5")

    # encoder = load_model("logs/desg_model/pths/encoder_epoch_7.h5", compile=False)
    # decoder = load_model("logs/desg_model/pths/decoder_epoch_2.h5",  compile=False)
    for i in range(1,14):
        cover_ = get_image2(f"D:\digitalwatermark\code\c.png")
        # cover_ = get_image(f"datasets/qrcodes/c10.png")
        # secret_ = get_image2(r"C:\Users\stfcomputer\Pictures\图像处理\cat.gif")
        # cover_ = get_image2(r"C:\Users\stfcomputer\Pictures\图像处理\tree.gif")
        secret_ = get_image2(f"datasets/test/{i}.png")  # 10397   16797

        from keras.utils import array_to_img
        import random



        # tecc =  get_image2(r"test/images/revolve30.png")


        wartermark_cover = encoder([secret_, cover_])
        # array_to_img(wartermark_cover[0]).save(f'./iiii/{random.random()}.png')
        wartermark_extract = decoder(wartermark_cover)
        # array_to_img(wartermark_extract[0]).save(f'./iiii/{random.random()}.png')
        fig, ax = plt.subplots(2, 2)
        fig.tight_layout()
        psnr = calculate_psnr1(cover_,wartermark_cover)
        print("psnr:",psnr)

        # array_to_img(cover_[0]).save('test/decoded_ctruet16801.png')
        # array_to_img(wartermark_cover[0]).save('test/decoded_cpredt16801.png')
        array_to_img(wartermark_extract[0]).save(f'./test/xiangsucha/sp{i}.png')
        array_to_img(secret_[0]).save(f'test/xiangsucha/st{i}.png')
        array_to_img(np.abs(secret_[0]-wartermark_extract[0]-5)).save(f'test/xiangsucha/sd{i}.png')



    # plt.subplot(221)
    # plt.imshow(cover_[0, :, :, 0])
    # plt.title('Qr[I]')
    #
    # plt.subplot(222)
    # plt.imshow(secret_[0, :, :, 0])
    # plt.title('Watermark[W]')
    #
    # plt.subplot(223)
    # plt.imshow(wartermark_cover[0, :, :, 0])
    # plt.title('qr_with_w')
    #
    # plt.subplot(224)
    # plt.imshow(wartermark_extract[0, :, :, 0])
    # plt.title('Extracted W')
    # plt.show(block=False)
    # plt.pause(1)
    # plt.close()
