from diffusers import StableDiffusionXLControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
from diffusers.utils import load_image
import torch
import os
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" # 保证程序cuda序号与实际cuda序号对应
os.environ['CUDA_VISIBLE_DEVICES'] = "1"
base_model_path = "/data/guyf/1/diff/examples/controlnet/model/aistable-diffusion-xl-base-1.0/stable-diffusion-xl-base-1.0"
controlnet_path = "/data/guyf/1/diff/examples/controlnet/model/demoire_5_13/checkpoint-10000/controlnet"

controlnet = ControlNetModel.from_pretrained(controlnet_path, torch_dtype=torch.float16)
pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
    base_model_path, controlnet=controlnet, torch_dtype=torch.float16,safety_checker=None,
)


pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)

pipe.enable_model_cpu_offload()
torch.backends.cudnn.benchmark = True

#           测试图片
control_image = load_image("/data/guyf/1/diff/examples/controlnet/data/image_test_part003_00000001_source.png")
# control_image = load_image("/data/guyf/1/diff/datasets/fill50k/moire/src_12396.png")
prompt = "demoire"

# generate image
generator = torch.manual_seed(0)
image = pipe(
    prompt, num_inference_steps=20, generator=generator, image=control_image,strength=0.9,#guidance_scale=9,
).images[0]
image.save("./output_5-12_09.png")#输出图片位置

controlnet = ControlNetModel.from_pretrained(controlnet_path, torch_dtype=torch.float16)
pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
    base_model_path, controlnet=controlnet, torch_dtype=torch.float16
)

# speed up diffusion process with faster scheduler and memory optimization
# pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
# # remove following line if xformers is not installed or when using Torch 2.0.
# pipe.enable_xformers_memory_efficient_attention()
# # memory optimization.
# pipe.enable_model_cpu_offload() strate 0.99
#
# control_image = load_image("./conditioning_image_1.png").resize((1024, 1024))
# prompt = "pale golden rod circle with old lace background"
#
# # generate image
# generator = torch.manual_seed(0)
# image = pipe(
#     prompt, num_inference_steps=20, generator=generator, image=control_image
# ).images[0]
# image.save("./output.png")