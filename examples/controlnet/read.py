# import cv2
# import numpy as np
# from PIL import Image
# from matplotlib import pyplot as plt
#
# img = cv2.imread(r'D:\study\dataset\demoire\clear\image_test_part001_00000002.png')
# b,g,r = cv2.split(img)
# img_rgb = cv2.merge([r,g,b])   # img_rgb = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)  上面两行也可以用它代替
# plt.imshow(img_rgb)
# plt.show()

# import cv2
# import numpy as np
#
#
# def laplacian_pyramid_mix(img1, img2):
#     # 构建第一张图的拉普拉斯金字塔
#     lp1 = [img1]
#     for _ in range(5):
#         img1 = cv2.pyrDown(img1)
#         lp1.append(img1)
#     lp1.reverse()
#
#     # 构建第二张图的拉普拉斯金字塔
#     lp2 = [img2]
#     for _ in range(5):
#         img2 = cv2.pyrDown(img2)
#         lp2.append(img2)
#     lp2.reverse()
#
#     # 合并拉普拉斯金字塔
#     mixed_lp = lp1[:4] + lp2[4:]
#
#     # 重建图像
#     result = mixed_lp[0]
#     for i in range(1, len(mixed_lp)):
#         result = cv2.pyrUp(result)
#         result = cv2.add(result, mixed_lp[i])
#
#     return result
#
#
# # 读取图片
# img1 = cv2.imread(r'D:\study\dataset\demoire\moire\image_test_part001_00000004.png')
# img2 = cv2.imread(r'D:\study\dataset\demoire\clear\image_test_part001_00000004.png')
#
# # 确保两张图片尺寸一致
# img1 = cv2.resize(img1, (img2.shape[1], img2.shape[0]))
#
# # 调用函数
# mixed_result = laplacian_pyramid_mix(img1, img2)
# cv2.imshow('Mixed Result', mixed_result)

# import cv2
# # 读取原图（无摩尔纹）和有摩尔纹的图
# # original = cv2.imread("original.png")
# # moire = cv2.imread("moire.png")
# original = cv2.imread(r'D:\study\dataset\demoire\moire\image_test_part001_00000002.png')
# moire = cv2.imread(r'D:\study\dataset\demoire\clear\image_test_part001_00000002.png')
#
# # 转换为 YCbCr 颜色空间
# original_ycbcr = cv2.cvtColor(original, cv2.COLOR_BGR2YCrCb)
# moire_ycbcr = cv2.cvtColor(moire, cv2.COLOR_BGR2YCrCb)
# def build_gaussian_pyramid(image, levels):
#     pyramid = [image]
#     for _ in range(levels-1):
#         image = cv2.pyrDown(image)
#         pyramid.append(image)
#     return pyramid
#
# def build_laplacian_pyramid(gaussian_pyramid):
#     laplacian_pyramid = []
#     for i in range(len(gaussian_pyramid)-1):
#         upsampled = cv2.pyrUp(gaussian_pyramid[i+1], dstsize=(gaussian_pyramid[i].shape[1], gaussian_pyramid[i].shape[0]))
#         laplacian = cv2.subtract(gaussian_pyramid[i], upsampled)
#         laplacian_pyramid.append(laplacian)
#     laplacian_pyramid.append(gaussian_pyramid[-1])  # 最后一层为高斯金字塔顶层
#     return laplacian_pyramid
#
# # 构建高斯金字塔（假设分3层）
# levels = 5
# original_gaussian = build_gaussian_pyramid(original_ycbcr, levels)
# moire_gaussian = build_gaussian_pyramid(moire_ycbcr, levels)
#
# # 构建拉普拉斯金字塔
# original_laplacian = build_laplacian_pyramid(original_gaussian)
# moire_laplacian = build_laplacian_pyramid(moire_gaussian)
# # 替换高斯金字塔的顶层（低频）
# moire_gaussian[-1] = original_gaussian[-1]  # 替换最后一层（低频）
#
# # 使用替换后的高斯金字塔重建图像
# reconstructed = moire_gaussian[-1]
# for i in range(len(moire_gaussian)-2, -1, -1):
#     reconstructed = cv2.pyrUp(reconstructed, dstsize=(moire_gaussian[i].shape[1], moire_gaussian[i].shape[0]))
#     reconstructed = cv2.add(reconstructed, moire_laplacian[i])  # 保留有摩尔纹的高频细节
# # 重建后的图像在YCbCr空间，需转回RGB
# reconstructed_ycbcr = reconstructed.astype('uint8')
# reconstructed_rgb = cv2.cvtColor(reconstructed_ycbcr, cv2.COLOR_YCrCb2BGR)
#
# # 保存调整后的图像
# cv2.imwrite(r'D:\study\dataset\demoire\blended_result1.png', reconstructed_rgb)

# import cv2
# import os
#
# # 设置路径
# moire_dir = r'D:\study\dataset\demoire\moire'
# clear_dir = r'D:\study\dataset\demoire\clear'
# output_dir = r'D:\study\dataset\demoire\moire1'
#
# # 创建输出目录
# os.makedirs(output_dir, exist_ok=True)
#
#
# def process_image(original_path, moire_path, output_path):
#     # 读取图像
#     original = cv2.imread(original_path)
#     moire = cv2.imread(moire_path)
#
#     # 转换到YCbCr颜色空间
#     original_ycbcr = cv2.cvtColor(original, cv2.COLOR_BGR2YCrCb)
#     moire_ycbcr = cv2.cvtColor(moire, cv2.COLOR_BGR2YCrCb)
#
#     def build_gaussian_pyramid(image, levels):
#         pyramid = [image]
#         for _ in range(levels - 1):
#             image = cv2.pyrDown(image)
#             pyramid.append(image)
#         return pyramid
#
#     def build_laplacian_pyramid(gaussian_pyramid):
#         laplacian_pyramid = []
#         for i in range(len(gaussian_pyramid) - 1):
#             upsampled = cv2.pyrUp(gaussian_pyramid[i + 1],
#                                   dstsize=(gaussian_pyramid[i].shape[1], gaussian_pyramid[i].shape[0]))
#             laplacian = cv2.subtract(gaussian_pyramid[i], upsampled)
#             laplacian_pyramid.append(laplacian)
#         laplacian_pyramid.append(gaussian_pyramid[-1])
#         return laplacian_pyramid
#
#     # 构建金字塔
#     levels = 5
#     original_gaussian = build_gaussian_pyramid(original_ycbcr, levels)
#     moire_gaussian = build_gaussian_pyramid(moire_ycbcr, levels)
#     moire_laplacian = build_laplacian_pyramid(moire_gaussian)
#
#     # 替换最后一层
#     moire_gaussian[-1] = original_gaussian[-1]
#
#     # 重建图像
#     reconstructed = moire_gaussian[-1]
#     for i in range(len(moire_gaussian) - 2, -1, -1):
#         reconstructed = cv2.pyrUp(reconstructed, dstsize=(moire_gaussian[i].shape[1], moire_gaussian[i].shape[0]))
#         reconstructed = cv2.add(reconstructed, moire_laplacian[i])
#
#     # 保存结果
#     reconstructed_rgb = cv2.cvtColor(reconstructed.astype('uint8'), cv2.COLOR_YCrCb2BGR)
#     cv2.imwrite(output_path, reconstructed_rgb)
#
#
# # 处理目录下所有图片
# for filename in os.listdir(moire_dir):
#     if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
#         moire_path = os.path.join(moire_dir, filename)
#         clear_path = os.path.join(clear_dir, filename)
#         output_path = os.path.join(output_dir, filename)
#
#         if os.path.exists(clear_path):
#             process_image(clear_path, moire_path, output_path)
#             print(f"已处理: {filename}")
#         else:
#             print(f"跳过: {filename} (未找到匹配的清晰图)")
#



# import cv2
#
# # 读取图像
# original = cv2.imread(r'D:\study\dataset\demoire\clear\image_test_part001_00000002.png')
# moire = cv2.imread(r'D:\study\dataset\demoire\clear\image_test_part001_00000002.png')
#
# # 转换到 YCrCb 颜色空间
# original_ycbcr = cv2.cvtColor(original, cv2.COLOR_BGR2YCrCb)
# moire_ycbcr = cv2.cvtColor(moire, cv2.COLOR_BGR2YCrCb)
#
# # 构建高斯金字塔和拉普拉斯金字塔
# def build_gaussian_pyramid(image, levels):
#     pyramid = [image]
#     for _ in range(levels-1):
#         image = cv2.pyrDown(image)
#         pyramid.append(image)
#     return pyramid
#
# def build_laplacian_pyramid(gaussian_pyramid):
#     laplacian_pyramid = []
#     for i in range(len(gaussian_pyramid)-1):
#         upsampled = cv2.pyrUp(gaussian_pyramid[i+1], dstsize=(gaussian_pyramid[i].shape[1], gaussian_pyramid[i].shape[0]))
#         laplacian = cv2.subtract(gaussian_pyramid[i], upsampled)
#         laplacian_pyramid.append(laplacian)
#     laplacian_pyramid.append(gaussian_pyramid[-1])  # 最后一层是高斯金字塔的顶层
#     return laplacian_pyramid
#
# levels = 3  # 金字塔层数
# x = 1       # 替换最后 x 层（可调整）
#
# original_gaussian = build_gaussian_pyramid(original_ycbcr, levels)
# moire_gaussian = build_gaussian_pyramid(moire_ycbcr, levels)
# moire_laplacian = build_laplacian_pyramid(moire_gaussian)
#
# # 替换最后 x 层（从最底层往上数 x 层）
# for lvl in range(levels - x, levels):
#     moire_gaussian[lvl] = original_gaussian[lvl]  # 完全替换
#     # 如果只想替换 Y 通道（亮度），可以改成：
#     # moire_gaussian[lvl][:, :, 0] = original_gaussian[lvl][:, :, 0]
#
# # 重建图像
# reconstructed = moire_gaussian[-1]  # 从最底层开始重建
# for i in range(levels-2, -1, -1):
#     reconstructed = cv2.pyrUp(reconstructed, dstsize=(moire_gaussian[i].shape[1], moire_gaussian[i].shape[0]))
#     reconstructed = cv2.add(reconstructed, moire_laplacian[i])  # 加上高频细节
#
# # 转换回 BGR 并保存
# reconstructed_rgb = cv2.cvtColor(reconstructed.astype('uint8'), cv2.COLOR_YCrCb2BGR)
# cv2.imwrite(r'D:\study\dataset\demoire\blended_result.png', reconstructed_rgb)
# import cv2
# import numpy as np
#
# # 读取图像
# original = cv2.imread(r'D:\study\dataset\demoire\clear\image_test_part001_00000002.png')
# moire = cv2.imread(r'D:\study\dataset\demoire\clear\image_test_part001_00000002.png')
#
# # 转换到YCrCb颜色空间
# original_ycbcr = cv2.cvtColor(original, cv2.COLOR_BGR2YCrCb)
# moire_ycbcr = cv2.cvtColor(moire, cv2.COLOR_BGR2YCrCb)
#
# # 构建高斯金字塔和拉普拉斯金字塔
# def build_gaussian_pyramid(image, levels):
#     pyramid = [image]
#     for _ in range(levels-1):
#         image = cv2.pyrDown(image)
#         pyramid.append(image)
#     return pyramid
#
# def build_laplacian_pyramid(gaussian_pyramid):
#     laplacian_pyramid = []
#     for i in range(len(gaussian_pyramid)-1):
#         upsampled = cv2.pyrUp(gaussian_pyramid[i+1], dstsize=(gaussian_pyramid[i].shape[1], gaussian_pyramid[i].shape[0]))
#         laplacian = cv2.subtract(gaussian_pyramid[i], upsampled)
#         laplacian_pyramid.append(laplacian)
#     laplacian_pyramid.append(gaussian_pyramid[-1])
#     return laplacian_pyramid
#
# levels = 3  # 金字塔层数
# x = 1       # 替换最后x层
#
# # 构建金字塔
# original_gaussian = build_gaussian_pyramid(original_ycbcr, levels)
# moire_gaussian = build_gaussian_pyramid(moire_ycbcr, levels)
# original_laplacian = build_laplacian_pyramid(original_gaussian)
# moire_laplacian = build_laplacian_pyramid(moire_gaussian)
#
# # 只替换Cr和Cb通道（通道1和2）
# for lvl in range(levels - x, levels):
#     # 对于拉普拉斯金字塔，只替换Cr和Cb通道
#     moire_laplacian[lvl][:, :, 1] = original_laplacian[lvl][:, :, 1]  # Cr通道
#     moire_laplacian[lvl][:, :, 2] = original_laplacian[lvl][:, :, 2]  # Cb通道
#
# # 重建图像（从拉普拉斯金字塔重建）
# reconstructed = moire_laplacian[-1]
# for i in range(levels-2, -1, -1):
#     reconstructed = cv2.pyrUp(reconstructed, dstsize=(moire_laplacian[i].shape[1], moire_laplacian[i].shape[0]))
#     reconstructed = cv2.add(reconstructed, moire_laplacian[i])
#
# # 转换回BGR颜色空间
# reconstructed_rgb = cv2.cvtColor(reconstructed.astype('uint8'), cv2.COLOR_YCrCb2BGR)
# cv2.imwrite(r'D:\study\dataset\demoire\blended_result_crcb.png', reconstructed_rgb)
import cv2
import numpy as np

#
# def ensure_even_dimensions(image):
#     """确保图像宽高为偶数（否则金字塔操作会报错）"""
#     h, w = image.shape[:2]
#     if h % 2 != 0:
#         image = image[:-1, :]  # 裁剪最后一行
#     if w % 2 != 0:
#         image = image[:, :-1]  # 裁剪最后一列
#     return image
#
#
# def build_laplacian_pyramid(image, levels):
#     """构建拉普拉斯金字塔（支持RGB图像）"""
#     pyramid = []
#     current = image.copy()
#     for _ in range(levels):
#         # 高斯模糊 + 降采样
#         blurred = cv2.GaussianBlur(current, (1, 1), 0)
#         down = cv2.pyrDown(blurred)
#         # 上采样并计算高频残差
#         up = cv2.pyrUp(down, dstsize=(current.shape[1], current.shape[0]))
#         laplacian = current - up
#         pyramid.append(laplacian)
#         current = down
#     pyramid.append(current)  # 最后一级是高斯金字塔的底层
#     return pyramid
#
#
# def reconstruct_from_pyramid(pyramid):
#     """从拉普拉斯金字塔重建图像"""
#     image = pyramid[-1]
#     for i in range(len(pyramid) - 2, -1, -1):
#         up = cv2.pyrUp(image, dstsize=(pyramid[i].shape[1], pyramid[i].shape[0]))
#         image = up + pyramid[i]
#     return image
#
#
# def laplacian_blend_rgb(original_rgb, moire_rgb, levels=5, replace_levels=3):
#     """
#     RGB图像的拉普拉斯金字塔混合
#     Args:
#         original_rgb: 原始图像 (H,W,3) RGB格式
#         moire_rgb: 摩尔纹图像 (H,W,3) RGB格式
#         levels: 金字塔层数
#         replace_levels: 替换的高频层数（1=仅边缘，2=边缘+中频）
#     Returns:
#         混合后的RGB图像 (H,W,3)
#     """
#     # 确保输入是RGB格式
#     original_bgr = cv2.cvtColor(original_rgb, cv2.COLOR_RGB2BGR)
#     moire_bgr = cv2.cvtColor(moire_rgb, cv2.COLOR_RGB2BGR)
#
#     # 对每个通道分别处理（BGR顺序）
#     blended_channels = []
#     for c in range(3):
#         # 构建金字塔
#         orig_pyramid = build_laplacian_pyramid(original_bgr[:, :, c], levels)
#         moire_pyramid = build_laplacian_pyramid(moire_bgr[:, :, c], levels)
#
#         # 替换高频层
#         for lvl in range(levels - replace_levels, levels):
#             moire_pyramid[lvl] = orig_pyramid[lvl]
#
#         # 重建通道
#         blended_channel = reconstruct_from_pyramid(moire_pyramid)
#         blended_channels.append(blended_channel)
#
#     # 合并通道并转回RGB
#     blended_bgr = np.stack(blended_channels, axis=-1)
#     blended_rgb = cv2.cvtColor(blended_bgr.astype(np.uint8), cv2.COLOR_BGR2RGB)
#     return blended_rgb
#
#
import cv2
import numpy as np
from PIL import Image
import numpy as np


def calculate_brightness(image_path):
    """计算图片的平均亮度（0~255，值越大越亮）"""
    img = Image.open(image_path).convert('L')  # 转为灰度图（'L'模式）
    pixels = np.array(img).flatten()  # 转为NumPy数组并展平
    return np.mean(pixels)


def compare_images(image1_path, image2_path):
    """比较两张图片的亮度"""
    brightness1 = calculate_brightness(image1_path)
    brightness2 = calculate_brightness(image2_path)

    print(f"图片1亮度: {brightness1:.2f}")
    print(f"图片2亮度: {brightness2:.2f}")

    if brightness1 > brightness2:
        print("图片1更亮")
    elif brightness1 < brightness2:
        print("图片2更亮")
    else:
        print("两张图片亮度相近")


# 示例调用


def compare_brightness(img1_path, img2_path):
    img1 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(img2_path, cv2.IMREAD_GRAYSCALE)

    avg1 = np.mean(img1)
    avg2 = np.mean(img2)

    print(f"图片1平均亮度: {avg1:.2f}, 图片2平均亮度: {avg2:.2f}")
    return "图片1更亮" if avg1 > avg2 else "图片2更亮"



def safe_laplacian_blend(original_rgb, moire_rgb, levels=3, replace_levels=1):
    """安全版金字塔混合（避免色斑）"""
    # 转BGR并裁剪至偶数尺寸
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

        # 构建摩尔纹金字塔（仅用于低频基底）
        gp_moire = [moire[..., c].astype(np.float32)]
        for _ in range(levels):
            gp_moire.append(cv2.pyrDown(gp_moire[-1]))

        # 重建：原始高频 + 摩尔纹低频
        layer = gp_orig[-1]  # 最底层直接用原始图像
        for lvl in range(levels - 1, -1, -1):
            layer = cv2.pyrUp(layer, dstsize=gp_orig[lvl].shape[::-1])
            if lvl >= levels - replace_levels:  # 替换高频层
                layer += (gp_orig[lvl] - cv2.pyrUp(gp_orig[lvl + 1], dstsize=gp_orig[lvl].shape[::-1]))
            else:  # 保留摩尔纹高频
                layer += (gp_moire[lvl] - cv2.pyrUp(gp_moire[lvl + 1], dstsize=gp_moire[lvl].shape[::-1]))
        blended[..., c] = np.clip(layer, 0, 255)

    return blended[..., ::-1].astype(np.uint8)  # BGR→RGB
# 示例用法
if __name__ == "__main__":
    # 读取图像（假设是RGB格式）
    original_rgb = cv2.cvtColor(cv2.imread("./data/image_test_part003_00000001_target.png"), cv2.COLOR_BGR2RGB)
    moire_rgb = cv2.cvtColor(cv2.imread(r'./data/image_test_part003_00000001_source.png'), cv2.COLOR_BGR2RGB)

    # 检查尺寸
    # original_rgb = ensure_even_dimensions(original_rgb)
    # moire_rgb = ensure_even_dimensions(moire_rgb)

    # 混合图像（替换1层高频=边缘）
    blended_rgb = safe_laplacian_blend(original_rgb, moire_rgb, levels=6, replace_levels=1)

    # 保存结果
    cv2.imwrite('./data/blended_result.png', cv2.cvtColor(blended_rgb, cv2.COLOR_RGB2BGR))
    print("混合完成！")
    # result = compare_brightness(r"D:\study\dataset\demoire\clear\image_test_part001_00000002.png", r"D:\study\dataset\demoire\blended_result.png")
    # print(result)
    compare_images("./data/image_test_part003_00000001_target.png",  "./data/blended_result.png")
# Load images (assuming they're tensors in [-1,1] range)

