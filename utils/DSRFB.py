from keras.layers import Conv2D, Add, Concatenate, Lambda
import tensorflow as tf

def pixel_shuffle(scale):
    return lambda x: tf.nn.depth_to_space(x, scale)

def pixel_unshuffle(scale):
    return lambda x: tf.nn.space_to_depth(x, scale)

def HRFMSBlock(inputs, filters=128, name='hrfms'):
    # c = inputs.shape[-1]
    # h = inputs.shape[1]
    # w = inputs.shape[2]
    #

    x = tf.keras.layers.Conv2D(48, (1, 1), padding='same', activation='relu')(inputs)
    x = tf.keras.layers.AveragePooling2D((4, 4), strides=(4, 4))(x)

    c = x.shape[-1]
    h = x.shape[1]
    w = x.shape[2]



    # 分通道 1/3 分支
    split = tf.split(x, num_or_size_splits=3, axis=-1)

    # 上支路：上采样 → 大卷积 → 下采样
    x1 = Conv2D(c//3, (1, 1), padding='same', activation='relu', name=name+'_x1_conv1')(split[0])
    x1 = Lambda(pixel_shuffle(2), name=name+'_x1_shuffle')(x1)
    x1 = Conv2D(c//12, (5, 5), padding='same', activation='relu', name=name+'_x1_conv2')(x1)
    x1 = Lambda(pixel_unshuffle(2), name=name+'_x1_unshuffle')(x1)

    # 中支路：中等卷积 → 下采样
    x2 = Conv2D(c//3, (1, 1), padding='same', activation='relu', name=name+'_x2_conv1')(split[1])
    x2 = Concatenate(axis=-1)([x1, x2])
    x2 = Conv2D(2*c//3, (3, 3), padding='same', activation='relu', name=name+'_x2_conv2')(x2)
    x2 = Lambda(pixel_unshuffle(2), name=name+'_x2_unshuffle')(x2)

    # 下支路：下采样 → 1x1 → 上采样
    x3 = Conv2D(c//3, (1, 1), padding='same', activation='relu', name=name+'_x3_conv1')(split[2])
    x3 = Lambda(pixel_unshuffle(2), name=name+'_x3_unshuffle')(x3)
    x3 = Concatenate(axis=-1)([x2, x3])
    x3 = Conv2D(4*c//3, (1, 1), padding='same', activation='relu', name=name+'_x3_conv2')(x3)
    x3 = Lambda(pixel_shuffle(2), name=name+'_x3_shuffle')(x3)

    # 融合 + 残差
    out = Conv2D(c, (1, 1), padding='same', activation='relu', name=name+'_final_conv')(x3)

    out = Add(name=name+'_residual')([x, out])

    out = tf.keras.layers.Conv2DTranspose(
        filters=48,
        kernel_size=(3, 3),
        strides=(4, 4),
        padding='same'
    )(out)  # → (b,64,64,48)
    out = tf.keras.layers.Conv2D(
        filters=3,
        kernel_size=(1, 1),
        padding='same',
        activation='sigmoid'
    )(out)

    return out