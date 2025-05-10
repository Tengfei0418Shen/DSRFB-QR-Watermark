import os
import shutil
from PIL import Image

# 指定源文件夹和目标文件夹
source_dir = r'D:\COCO2017\test2017'  # 替换为你的源文件夹路径
target_dir = 'datasets/train/'  # 替换为你的目标文件夹路径
resize_dim = (64, 64)  # 设置目标大小为 64x64

# 如果目标文件夹不存在，则创建该文件夹
if not os.path.exists(target_dir):
    os.makedirs(target_dir)
count = 0
# 遍历源文件夹中的所有文件
for filename in os.listdir(source_dir):
    if count==50000:
        break
    # 构建完整的文件路径
    source_path = os.path.join(source_dir, filename)

    # 确保它是一个文件并且是图片
    if os.path.isfile(source_path) and filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
        try:
            count+=1
            # 打开图片
            with Image.open(source_path) as img:
                # 调整大小
                img_resized = img.resize(resize_dim)

                # 构建目标文件路径
                target_path = os.path.join(target_dir, f'{count}.png')

                # 保存调整大小后的图片到目标文件夹
                img_resized.save(target_path)
                print(f"已处理: {target_path}")
        except Exception as e:
            print(f"处理 {filename} 时发生错误: {e}")
