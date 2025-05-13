import os
import shutil

# 定义路径
source_dir = r"D:\study\dataset\demoire\image_test_part007"
moire_dir = r"D:\study\dataset\demoire\moire"
clear_dir = r"D:\study\dataset\demoire\clear"

# 确保目标文件夹存在
os.makedirs(moire_dir, exist_ok=True)
os.makedirs(clear_dir, exist_ok=True)

# 遍历源文件夹
for filename in os.listdir(source_dir):
    # 处理_source文件
    if "_source" in filename:
        source_path = os.path.join(source_dir, filename)
        # 生成新文件名（去掉_source）
        new_filename = filename.replace("_source", "")
        dest_path = os.path.join(moire_dir, new_filename)
        shutil.move(source_path, dest_path)
        print(f"Moved {filename} to moire folder as {new_filename}")

    # 处理_target文件
    elif "_target" in filename:
        source_path = os.path.join(source_dir, filename)
        # 生成新文件名（去掉_target）
        new_filename = filename.replace("_target", "")
        dest_path = os.path.join(clear_dir, new_filename)
        shutil.move(source_path, dest_path)
        print(f"Moved {filename} to clear folder as {new_filename}")

print("文件分类移动完成！")