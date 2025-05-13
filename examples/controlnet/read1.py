import cv2
import numpy as np
import os
from tqdm import tqdm  # 进度条工具（可选，安装：pip install tqdm）

def safe_laplacian_blend(original_rgb, moire_rgb, levels=3, replace_levels=1):
    """安全版拉普拉斯金字塔混合（避免色偏）"""
    # 转换BGR并裁剪至偶数尺寸
    original = original_rgb[..., ::-1]  # RGB→BGR
    moire = moire_rgb[..., ::-1]
    h, w = original.shape[:2]
    original = original[:h // 2 * 2, :w // 2 * 2]
    moire = moire[:h // 2 * 2, :w // 2 * 2]

    blended = np.zeros_like(original, dtype=np.float32)
    for c in range(3):  # 分通道处理
        # 构建原始图像金字塔
        gp_orig = [original[..., c].astype(np.float32)]
        for _ in range(levels):
            gp_orig.append(cv2.pyrDown(gp_orig[-1]))

        # 构建摩尔纹图像金字塔（仅用于低频基底）
        gp_moire = [moire[..., c].astype(np.float32)]
        for _ in range(levels):
            gp_moire.append(cv2.pyrDown(gp_moire[-1]))

        # 重建：原始高频 + 摩尔纹低频
        layer = gp_orig[-1]
        for lvl in range(levels - 1, -1, -1):
            layer = cv2.pyrUp(layer, dstsize=gp_orig[lvl].shape[::-1])
            if lvl >= levels - replace_levels:  # 替换高频层
                layer += (gp_orig[lvl] - cv2.pyrUp(gp_orig[lvl + 1], dstsize=gp_orig[lvl].shape[::-1]))
            else:  # 保留摩尔纹高频
                layer += (gp_moire[lvl] - cv2.pyrUp(gp_moire[lvl + 1], dstsize=gp_moire[lvl].shape[::-1]))
        blended[..., c] = np.clip(layer, 0, 255)

    return blended[..., ::-1].astype(np.uint8)  # BGR→RGB

def batch_process(clear_dir, moire_dir, output_dir, levels=6, replace_levels=1):
    """批量处理文件夹中的图片"""
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    # 获取匹配的图片文件名（假设clear和moire中的图片名一一对应）
    clear_files = sorted([f for f in os.listdir(clear_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])
    moire_files = sorted([f for f in os.listdir(moire_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])

    # 检查文件数量是否一致
    if len(clear_files) != len(moire_files):
        print("警告：clear和moire文件夹中的图片数量不一致！")

    # 处理每对图片
    for clear_file, moire_file in tqdm(zip(clear_files, moire_files), total=len(clear_files)):
        try:
            # 读取图片（假设是RGB格式）
            clear_path = os.path.join(clear_dir, clear_file)
            moire_path = os.path.join(moire_dir, moire_file)
            original_rgb = cv2.cvtColor(cv2.imread(clear_path), cv2.COLOR_BGR2RGB)
            moire_rgb = cv2.cvtColor(cv2.imread(moire_path), cv2.COLOR_BGR2RGB)

            # 混合图像
            blended_rgb = safe_laplacian_blend(original_rgb, moire_rgb, levels, replace_levels)

            # 保存结果（保持原文件名）
            output_path = os.path.join(output_dir, clear_file)
            cv2.imwrite(output_path, cv2.cvtColor(blended_rgb, cv2.COLOR_RGB2BGR))
        except Exception as e:
            print(f"处理 {clear_file} 和 {moire_file} 时出错：{e}")

if __name__ == "__main__":
    # 输入输出路径配置
    clear_dir = r"D:\study\dataset\demoire\clear"    # 清晰图片目录
    moire_dir = r"D:\study\dataset\demoire\moire"   # 摩尔纹图片目录
    output_dir = r"D:\study\dataset\demoire\moire1" # 输出目录

    # 批量处理（可调整金字塔参数）
    batch_process(clear_dir, moire_dir, output_dir, levels=6, replace_levels=1)
    print(f"处理完成！结果已保存到 {output_dir}")