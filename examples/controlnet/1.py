# from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
# from diffusers.utils import load_image
# import torch
#
# base_model_path = "/data/guyf/1/diff/examples/controlnet/model/stable-diffusion-v1-5"
# controlnet_path = "/data/guyf/1/diff/examples/controlnet/model/checkpoint-6000/controlnet"
#
# controlnet = ControlNetModel.from_pretrained(controlnet_path, torch_dtype=torch.float16)
# pipe = StableDiffusionControlNetPipeline.from_pretrained(
#     base_model_path, controlnet=controlnet, torch_dtype=torch.float16
# )
#
# # speed up diffusion process with faster scheduler and memory optimization
# pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
# # remove following line if xformers is not installed or when using Torch 2.0.
# pipe.enable_xformers_memory_efficient_attention()
# # memory optimization.
# pipe.enable_model_cpu_offload()
#
# control_image = load_image("./conditioning_image_1.png")
# prompt = "pale golden rod circle with old lace background"
#
# # generate image
# generator = torch.manual_seed(0)
# image = pipe(
#     prompt, num_inference_steps=20, generator=generator, image=control_image
# ).images[0]
# image.save("./output.png")
import os
import json
import re

# 定义文件夹路径
moire_folder = "/data/guyf/1/diff/datasets/fill50k/moire"
clear_folder = "/data/guyf/1/diff/datasets/fill50k/clear"

# 获取两个文件夹中的文件列表
moire_files = set(f for f in os.listdir(moire_folder) if f.endswith('.png'))
clear_files = set(f for f in os.listdir(clear_folder) if f.endswith('.png'))

# 找出两个文件夹中共同的文件名
common_files = moire_files & clear_files
print("Sample moire files:", [f for f in moire_files if 'tar_' in f][:5])
print("Sample clear files:", [f for f in clear_files if 'src_' in f][:5])

# 4. 提取数字的函数（严格匹配 `_数字.png`）
def extract_number(filename):
    match = re.search(r'_(\d+)\.png$', filename)  # 只匹配 _数字.png
    return match.group(1) if match else None

# 5. 提取 tar_ 和 src_ 文件（严格匹配前缀）
tar_files = {extract_number(f): f for f in moire_files if f.startswith('tar_')}
src_files = {extract_number(f): f for f in clear_files if f.startswith('src_')}

print("Tar files found:", tar_files)
print("Src files found:", src_files)

# 6. 匹配数字相同的文件对
matched_pairs = []
for num in set(tar_files.keys()) & set(src_files.keys()):
    matched_pairs.append((tar_files[num], src_files[num]))

print("Matched pairs:", matched_pairs)

# 生成 JSON Lines 文件（.jsonl）
with open(r"D:\study\dataset\demoire\clear\train_512.jsonl", "w") as f:
    for filename in sorted(common_files):
        json_record = {
            "text": f"demoire", "image": f"moire/{filename}",
            "conditioning_image": f"clear/{filename}"
        }
        f.write(json.dumps(json_record) + "\n")

    for tar_file, scr_file in sorted(matched_pairs):
        print(tar_file, scr_file)
        json_record = {
            "text": "demoire",
            "image": f"moire/{tar_file}",
            "conditioning_image": f"clear/{scr_file}"
        }
        f.write(json.dumps(json_record) + "\n")

print(f"成功生成 train.jsonl 文件，共包含 {len(common_files)} 对图片。")

