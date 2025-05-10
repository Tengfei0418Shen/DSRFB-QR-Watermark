import math
import os

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from keras import Model
from keras.layers import *
from tqdm import tqdm
import keras.backend as K
import tensorflow as tf
import numpy as np
import tensorflow_addons as tfa

# Variable used to weight the losses of the secret and cover images (See paper for more details)
beta = 1.0

def rev_loss(s_true, s_pred):
    extraction_loss = K.sum(K.square(s_true - s_pred))
    return beta * extraction_loss

# # Loss for the full model (self-supervised)
def full_loss(y_true, y_pred):
    s_true, c_true = y_true[0], y_true[1]
    c_pred = y_pred
    s_loss = K.sum(K.square(s_true - c_pred))
    c_loss = K.sum(K.square(c_true - c_pred))
    return (beta - 0.8) * s_loss + c_loss


def multiply_255(x):
    return x * 255.0


def divide_255(x):
    return x / 255.0


# Encoder Model
def make_encoder(input_size):
    input_S = Input(shape=(input_size))
    input_C = Input(shape=(input_size))

    # Apply noise reduction (e.g., Gaussian smoothing)
    x = Conv2D(64, (3, 3), padding='same', activation='relu')(input_S)
    x = Conv2D(64, (3, 3), padding='same', activation='relu')(x)

    # Encoder layers (similar to original design but simplified)
    x3 = Conv2D(50, (3, 3), strides=(1, 1), padding="same", activation="relu")(x)
    x4 = Conv2D(10, (4, 4), strides=(1, 1), padding="same", activation="relu")(x)
    x5 = Conv2D(5, (5, 5), strides=(1, 1), padding="same", activation="relu")(x)
    x = concatenate([x3, x4, x5])

    x = concatenate([input_C, x])

    # Hiding network
    x = Conv2D(50, (3, 3), strides=(1, 1), padding="same", activation="relu")(x)
    x = Conv2D(50, (3, 3), strides=(1, 1), padding="same", activation="relu")(x)

    output_Cprime = Conv2D(3, (3, 3), strides=(1, 1), padding="same", activation="relu")(x)

    return Model(inputs=[input_S, input_C], outputs=output_Cprime, name="Encoder")


# Decoder Model
def make_decoder(input_size=(64, 64, 3), block_size=4):
    # Reveal network
    reveal_input = Input(shape=(input_size))

    # Adding Gaussian noise with 0.01 standard deviation.
    input_with_noise = GaussianNoise(0.01, name="output_C_noise")(reveal_input)


    x3 = Conv2D(
        50,
        (3, 3),
        strides=(1, 1),
        padding="same",
        activation="relu",
        name="conv_rev0_3x3",
    )(input_with_noise)
    x4 = Conv2D(
        10,
        (4, 4),
        strides=(1, 1),
        padding="same",
        activation="relu",
        name="conv_rev0_4x4",
    )(input_with_noise)
    x5 = Conv2D(
        5,
        (5, 5),
        strides=(1, 1),
        padding="same",
        activation="relu",
        name="conv_rev0_5x5",
    )(input_with_noise)
    x = concatenate([x3, x4, x5])

    x3 = Conv2D(
        50,
        (3, 3),
        strides=(1, 1),
        padding="same",
        activation="relu",
        name="conv_rev1_3x3",
    )(x)
    x4 = Conv2D(
        10,
        (4, 4),
        strides=(1, 1),
        padding="same",
        activation="relu",
        name="conv_rev1_4x4",
    )(x)
    x5 = Conv2D(
        5,
        (5, 5),
        strides=(1, 1),
        padding="same",
        activation="relu",
        name="conv_rev1_5x5",
    )(x)
    x = concatenate([x3, x4, x5])

    x3 = Conv2D(
        50,
        (3, 3),
        strides=(1, 1),
        padding="same",
        activation="relu",
        name="conv_rev2_3x3",
    )(x)
    x4 = Conv2D(
        10,
        (4, 4),
        strides=(1, 1),
        padding="same",
        activation="relu",
        name="conv_rev2_4x4",
    )(x)
    x5 = Conv2D(
        5,
        (5, 5),
        strides=(1, 1),
        padding="same",
        activation="relu",
        name="conv_rev2_5x5",
    )(x)
    x = concatenate([x3, x4, x5])

    x3 = Conv2D(
        50,
        (3, 3),
        strides=(1, 1),
        padding="same",
        activation="relu",
        name="conv_rev3_3x3",
    )(x)
    x4 = Conv2D(
        10,
        (4, 4),
        strides=(1, 1),
        padding="same",
        activation="relu",
        name="conv_rev3_4x4",
    )(x)
    x5 = Conv2D(
        5,
        (5, 5),
        strides=(1, 1),
        padding="same",
        activation="relu",
        name="conv_rev3_5x5",
    )(x)
    x = concatenate([x3, x4, x5])

    x3 = Conv2D(
        50,
        (3, 3),
        strides=(1, 1),
        padding="same",
        activation="relu",
        name="conv_rev4_3x3",
    )(x)
    x4 = Conv2D(
        10,
        (4, 4),
        strides=(1, 1),
        padding="same",
        activation="relu",
        name="conv_rev4_4x4",
    )(x)
    x5 = Conv2D(
        5,
        (5, 5),
        strides=(1, 1),
        padding="same",
        activation="relu",
        name="conv_rev5_5x5",
    )(x)
    x = concatenate([x3, x4, x5])
    output_Sprime = Conv2D(
        3, (3, 3), strides=(1, 1), padding="same", activation="relu", name="output_S"
    )(x)

    # if not fixed:
    return Model(inputs=reveal_input, outputs=output_Sprime, name="Decoder")


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

def psnr_(img1, img2):
    mse = np.mean((img1 / 255.0 - img2 / 255.0) ** 2)
    if mse < 1e-10:
        return 100
    psnr2 = 20 * math.log10(1 / math.sqrt(mse))
    return psnr2

def get_cover_images(i, batch_size):
    temp=[]
    for j in range(i,i+batch_size):
        with Image.open(f"datasets/qrcodes/c{j+1}.png").convert("L") as img:
            temp.append(img)
    temp = tf.convert_to_tensor(np.array(temp), dtype='float32')
    temp = tf.expand_dims(temp, axis=-1)
    temp = tf.repeat(temp, repeats=3, axis=-1)
    return temp

def get_watermark_images(i, batch_size):
    temp=[]
    for j in range(i,i+batch_size):
        with Image.open(f"datasets/train/{j+1}.png").convert("L") as img:
            temp.append(img)

    temp = tf.convert_to_tensor(np.array(temp), dtype='float32')
    temp = tf.expand_dims(temp, axis=-1)
    temp = tf.repeat(temp, repeats=3, axis=-1)
    return temp

exp_id = 'desg_model_test'
num_of_train = 40000
directory = os.path.join('logs', exp_id)
if not os.path.exists(directory):
    os.makedirs(directory)
    os.makedirs(os.path.join(directory, "images"))
    os.makedirs(os.path.join(directory, "weights"))
    os.makedirs(os.path.join(directory, "pths"))
    f = open(os.path.join(directory, "train_info.txt"),"w")
    f.close()
# Training Setup

def train_model(encoder, decoder, epochs=10, batch_size=32, learning_rate=0.001):
    # Compile the models
    encoder.compile(optimizer=tf.keras.optimizers.Adam(learning_rate), loss=full_loss)
    decoder.compile(optimizer=tf.keras.optimizers.Adam(learning_rate), loss=rev_loss)

    # decoder.load_weights(directory + "/weights/decoder_epoch_19.h5")
    # encoder.load_weights(directory + "/weights/encoder_epoch_19.h5")

    # Create a custom training loop
    print('train....')
    for epoch in range(epochs):

        print(f"Epoch {epoch + 1}/{epochs}")
        total_loss = 0
        total_rev_loss = 0
        total_psnr = []
        # Train the model in batches
        # for i in tqdm(range(0, num_samples, batch_size), desc=f"Training Epoch {epoch + 1}", ncols=100):
        for i in tqdm(range(0, num_of_train, batch_size), desc=f"Training Epoch {epoch + 1}", ncols=100):
            # Get the batch
            batch_C = get_cover_images(i,batch_size)
            # batch_S = get_cover_images(i,batch_size)
            batch_S = get_watermark_images(i,batch_size)
            # batch_C = get_watermark_images(i,batch_size)
            with tf.device('/GPU:0'):  #
                # Compute the reveal loss separately (decoding the cover images)
                with tf.GradientTape() as tape:
                    output_Cprime = encoder([batch_S, batch_C])
                    loss = full_loss([batch_S, batch_C], output_Cprime)
                    total_loss += loss
                    # print("Trainable variables:", decoder.trainable_variables)
                    gradients = tape.gradient(loss, encoder.trainable_variables)
                    encoder.optimizer.apply_gradients(zip(gradients, encoder.trainable_variables))
                with tf.GradientTape() as tape:
                    output_Sprime = decoder(output_Cprime)
                # Compute gradients for decoder
                    rev_loss_val = rev_loss(batch_S, output_Sprime)
                    total_rev_loss += rev_loss_val

                    # print("Trainable variables:", decoder.trainable_variables)
                    gradients = tape.gradient(rev_loss_val, decoder.trainable_variables)
                    decoder.optimizer.apply_gradients(zip(gradients, decoder.trainable_variables))

                for t in range(batch_size):
                    # total_psnr.append(calculate_psnr(batch_C[t], output_Cprime[t]))
                    total_psnr.append(psnr_(batch_C[t], output_Cprime[t]))

                tf.keras.backend.clear_session()

                if i % 80 == 0:
                    fig, ax = plt.subplots(2, 4)
                    fig.tight_layout()
                    plt.subplot(241)
                    plt.imshow(batch_C[0, :, :, 0], cmap='gray')
                    plt.title('Qr[I]')

                    plt.subplot(242)
                    plt.imshow(batch_S[0, :, :, 0], cmap='gray')
                    plt.title('Watermark[W]')

                    plt.subplot(245)
                    plt.imshow(output_Cprime[0, :, :, 0] * 255, cmap='gray')
                    plt.title('qr_with_w')

                    plt.subplot(246)
                    plt.imshow(output_Sprime[0, :, :, 0] * 255, cmap='gray')
                    plt.title('Extracted W')

                    plt.subplot(243)
                    plt.imshow(batch_C[1, :, :, 0], cmap='gray')
                    plt.title('Qr[I]')

                    plt.subplot(244)
                    plt.imshow(batch_S[1, :, :, 0], cmap='gray')
                    plt.title('Watermark[W]')

                    plt.subplot(247)
                    plt.imshow(output_Cprime[1, :, :, 0] * 255, cmap='gray')
                    plt.title('qr_with_w')

                    plt.subplot(248)
                    plt.imshow(output_Sprime[1, :, :, 0] * 255, cmap='gray')
                    plt.title('Extracted W')
                    plt.savefig(os.path.join(os.path.join(directory, 'images/'), f'train_epoch{epoch}_{i}.png'))
                    plt.show(block=False)
                    plt.pause(1)
                    plt.close()

        avg_psnr = np.mean(total_psnr)
        print("total_length",len(total_psnr))
        # 打印最大的前五个值
        total_psnr = np.array(total_psnr)
        sorted_indices = np.argsort(-np.array(total_psnr))

        # 打印最大的前五个值
        top_five_values = total_psnr[sorted_indices[:5]]
        # print("最大的前五个值：", top_five_values)

        # 打印对应的索引
        top_five_indices = sorted_indices[:5]
        # print("对应的索引：", top_five_indices)


        print(f"Loss: {total_loss / num_of_train}, Reveal Loss: {total_rev_loss / num_of_train} , PSNR:{avg_psnr}")

        with open(os.path.join(directory, 'train_info.txt'), 'a') as f:
            f.write(f"EPOCH {epoch+1}  Loss: {total_loss / num_of_train}, Reveal Loss: {total_rev_loss / num_of_train}, PSNR:{avg_psnr} {top_five_indices} {top_five_values}\n")
            f.close()
        # 保存模型检查点（如果需要）
        # Save the model at the end of each epoch
        encoder.save(os.path.join(directory, f'pths/encoder_epoch_{epoch + 1}.h5'))
        encoder.save_weights(os.path.join(directory, f'weights/encoder_epoch_{epoch + 1}.h5'))
        decoder.save(os.path.join(directory, f'pths/decoder_epoch_{epoch + 1}.h5'))
        decoder.save_weights(os.path.join(directory, f'weights/decoder_epoch_{epoch + 1}.h5'))

        # Optionally, plot loss curves and save them
        plt.plot(total_loss, label='Loss')
        plt.plot(total_rev_loss, label='Reveal Loss')
        plt.legend()
        plt.savefig(os.path.join(directory, f'images/loss_epoch_{epoch + 1}.png'))
        plt.close()


# Example of using the function with mock data
input_size = (64, 64, 3)  # Example input size
encoder = make_encoder(input_size)

decoder = make_decoder(input_size)

print('loading data.....')

# 确保 TensorFlow 会使用 GPU
physical_devices = tf.config.list_physical_devices('GPU')
print(physical_devices)
if physical_devices:
    print("Using GPU")
    tf.config.set_visible_devices(physical_devices[0], 'GPU')  # 选择第一个GPU
else:
    print("No GPU found, using CPU.")

# encoder, decoder, _ = make_model(input_size)

train_model(encoder, decoder, epochs=21, batch_size=32)
