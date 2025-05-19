from keras.layers import *
from keras.models import Model
from .noises import *
from .loss import *


def multiply_255(x):
    return x * 255.0


def divide_255(x):
    return x / 255.0


# Encoder Model
def make_encoder(input_size):
    input_S = Input(shape=(input_size))
    input_C = Input(shape=(input_size))

    # Apply noise reduction (e.g., Gaussian smoothing)
    # x = Conv2D(64, (3, 3), padding='same', activation='relu')(input_S)
    # x = Conv2D(64, (3, 3), padding='same', activation='relu')(x)

    # Encoder layers (similar to original design but simplified)
    x3 = Conv2D(50, (3, 3), strides=(1, 1), padding="same", activation="relu")(input_S)
    x4 = Conv2D(10, (4, 4), strides=(1, 1), padding="same", activation="relu")(input_S)
    x5 = Conv2D(5, (5, 5), strides=(1, 1), padding="same", activation="relu")(input_S)
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

    # input_with_noise = GaussianNoise(stddev=0.003, name='rounding_noise')(attacked_Iw)

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