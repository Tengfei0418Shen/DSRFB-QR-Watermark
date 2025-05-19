import keras.backend as K


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



def watermark_visibility_loss(y_true, y_pred):
    """
    计算水印图像的不可见性损失，鼓励水印图像在嵌入后尽可能不可见。
    y_true: 原始水印图像
    y_pred: 解码器恢复的水印图像
    """
    return K.mean(K.square(y_true - y_pred))  # L2损失


def cover_quality_loss(y_true, y_pred):
    """
    计算封面图像的质量损失，确保嵌入水印后的封面图像与原封面图像的差异最小。
    y_true: 原始封面图像（QR码）
    y_pred: 嵌入水印后的封面图像（包括水印）
    """
    return K.mean(K.square(y_true - y_pred))  # L2损失


def total_loss(y_true_S, y_pred_S, y_true_C, y_pred_C, lambda_w=1.0, lambda_c=1.0):
    """
    总损失函数，包括水印不可见性损失和封面图像质量损失。
    y_true_S: 原始水印图像
    y_pred_S: 解码器恢复的水印图像
    y_true_C: 原始封面图像（QR码）
    y_pred_C: 嵌入水印后的封面图像（QR码）
    lambda_w: 水印不可见性损失的权重
    lambda_c: 封面图像质量损失的权重
    """
    watermark_loss = watermark_visibility_loss(y_true_S, y_pred_S)
    cover_loss = cover_quality_loss(y_true_C, y_pred_C)

    total_loss = lambda_w * watermark_loss + lambda_c * cover_loss
    return total_loss
