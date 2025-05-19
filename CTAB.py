import keras.backend as K
from keras.layers import GlobalAveragePooling2D, GlobalMaxPooling2D, Dense, Activation, Reshape, Multiply, Conv2D, \
    Concatenate, Flatten, Add
import tensorflow as tf


def TCAMBlock(x, reduction=8):
    channel_dim = K.int_shape(x)[-1]

    # -------- 通道注意力 --------
    avg_pool = GlobalAveragePooling2D()(x)
    max_pool = GlobalMaxPooling2D()(x)

    # ⚠ 确保 avg_pool / max_pool 是二维的
    avg_pool = Reshape((1, channel_dim))(avg_pool)
    max_pool = Reshape((1, channel_dim))(max_pool)

    shared_dense = Dense(channel_dim // reduction, activation='relu')
    mlp_avg = Dense(channel_dim)(shared_dense(avg_pool))
    mlp_max = Dense(channel_dim)(shared_dense(max_pool))

    # ⚠ Flatten 回 1D
    mlp_avg = Flatten()(mlp_avg)
    mlp_max = Flatten()(mlp_max)

    channel_attn = Activation('sigmoid')(Add()([mlp_avg, mlp_max]))
    channel_attn = Reshape((1, 1, channel_dim))(channel_attn)
    x = Multiply()([x, channel_attn])

    # -------- Triplet-style 空间注意力 --------
    def spatial_attention(inp):
        avg = tf.reduce_mean(inp, axis=-1, keepdims=True)
        max_ = tf.reduce_max(inp, axis=-1, keepdims=True)
        concat = Concatenate(axis=-1)([avg, max_])
        return Conv2D(1, kernel_size=7, padding='same', activation='sigmoid')(concat)

    attn_hw = spatial_attention(x)
    attn_wh = tf.transpose(spatial_attention(tf.transpose(x, [0, 2, 1, 3])), [0, 2, 1, 3])
    attn_ch = tf.transpose(spatial_attention(tf.transpose(x, [0, 3, 2, 1])), [0, 3, 2, 1])

    x = Multiply()([x, attn_hw, attn_wh, attn_ch])

    return x