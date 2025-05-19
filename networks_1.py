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

    x = Conv2D(32, (3, 3), padding='same', activation='relu', trainable=False, name='scramble_layer')(input_S)

    # Apply noise reduction (e.g., Gaussian smoothing)
    x = Conv2D(64, (3, 3), padding='same', activation='relu')(x)
    x = Conv2D(64, (3, 3), padding='same', activation='relu')(x)

    # Encoder layers (similar to original design but simplified)
    x3 = Conv2D(50, (3, 3), strides=(1, 1), padding="same", activation="relu")(x)
    x4 = Conv2D(10, (4, 4), strides=(1, 1), padding="same", activation="relu")(x)
    x5 = Conv2D(5, (5, 5), strides=(1, 1), padding="same", activation="relu")(x)
    x = concatenate([x3, x4, x5], axis=-1)

    x = concatenate([input_C, x], axis=-1)

    # Hiding network
    x = Conv2D(50, (3, 3), strides=(1, 1), padding="same", activation="relu")(x)
    x = Conv2D(50, (3, 3), strides=(1, 1), padding="same", activation="relu")(x)

    output_Cprime = Conv2D(1, (3, 3), strides=(1, 1), padding="same", activation="relu")(x)

    return Model(inputs=[input_S, input_C], outputs=output_Cprime, name="Encoder")


# Decoder Model
def make_decoder(input_size, block_size=4):
    # Reveal network
    reveal_input = Input(shape=(input_size))

    # Adding Gaussian noise with 0.01 standard deviation.
    # input_with_noise = GaussianNoise(0.01, name="output_C_noise")(reveal_input)

    no_attacked = x = reveal_input

    ##################### Noise Attack  #############################
    noise_std = 3.0
    noise_attacked = Activation(multiply_255)(reveal_input)
    noise_attacked = GaussianNoise(stddev=noise_std, name='guassian_noise_attack')(noise_attacked)
    noise_attacked = Activation(divide_255)(noise_attacked)

    ###################  Salt and Pepper Attack ############################
    salt_pepper_attacked = Lambda(salt_pepper_noise, arguments={'salt_ratio': 0.04}, name='salt_pepper_attack')(
        reveal_input)

    rotation_attacked = Lambda(rotate_image11, name='rotation_attack')(reveal_input)

    #####################    smoothing_attack  #######################

    smoothing_layer = Conv2D(
        filters=1,  # 保持与输入通道数一致
        kernel_size=(3, 3),
        padding='same',
        name='smoothing_attack',
        use_bias=False,
        trainable=False
    )
    mean_attacked = smoothing_layer(x)

    mean_attacked = Lambda(
        lambda x: tf.nn.space_to_depth(x, block_size=block_size),
        name='mean_attack_space2depth'
    )(mean_attacked)
    # 恢复空间并减少通道数
    mean_attacked = Lambda(
        lambda x: tf.nn.depth_to_space(x, block_size=block_size),
        name='mean_attack_restore'
    )(mean_attacked)

    #####################    sharpenning(edge)_attak  #######################
    # mean = smoothing_layer(x)
    # mean = Lambda(multiply_scalar, arguments={'scalar': -1.0})(mean)
    # x2 = Lambda(multiply_scalar, arguments={'scalar': 1.0})(x)
    #
    # sharpenning_attacked = Add(name='sharpenning_subtract')([x2, mean])
    # sharpenning_attacked = Lambda(tf.nn.space_to_depth,arguments={'block_size': block_size},
    #                                      name='sharpening_attack_space2depth')(sharpenning_attacked)



    attack_list = [salt_pepper_attacked, noise_attacked, mean_attacked, rotation_attacked,no_attacked]
    attack_prob = None
    attacked_Iw = Lambda(random_switch, arguments={'prob': attack_prob}, name='Random_selection_of_attacks')(
        attack_list)
    input_with_noise = GaussianNoise(stddev=0.003, name='rounding_noise')(attacked_Iw)
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

    x = concatenate([x3, x4, x5], axis=-1)
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
    x = concatenate([x3, x4, x5], axis=-1)

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
    x = concatenate([x3, x4, x5], axis=-1)

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
    x = concatenate([x3, x4, x5], axis=-1)

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
    x = concatenate([x3, x4, x5], axis=-1)

    x = Conv2D(32, (3, 3), padding='same', activation='relu', trainable=False, name='descramble_layer')(x)

    output_Sprime = Conv2D(
        1, (3, 3), strides=(1, 1), padding="same", activation="relu", name="output_S"
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
        inputs=[input_S, input_C], outputs=concatenate([output_Sprime, output_Cprime], axis=-1)
    )
    autoencoder.compile(optimizer=tf.keras.optimizers.Adam(0.001), loss=full_loss)

    return encoder, decoder, autoencoder