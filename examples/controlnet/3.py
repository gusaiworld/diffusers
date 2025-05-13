from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
from diffusers.utils import load_image
import torch

base_model_path = "/data/guyf/1/diff/examples/controlnet/model/stable-diffusion-v1-5"
controlnet_path = "/data/guyf/1/diff/examples/controlnet/model/demoire_4_30_bhd/checkpoint-3000/controlnet"

controlnet = ControlNetModel.from_pretrained(controlnet_path, torch_dtype=torch.float16)
pipe = StableDiffusionControlNetPipeline.from_pretrained(
    base_model_path, controlnet=controlnet, torch_dtype=torch.float16,safety_checker=None,
)

# speed up diffusion process with faster scheduler and memory optimization
pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
# remove following line if xformers is not installed or when using Torch 2.0.
# pipe.enable_xformers_memory_efficient_attention()
# memory optimization.
pipe.enable_model_cpu_offload()
torch.backends.cudnn.benchmark = True
# a='image_test_part002_00008452.png'
# patth=os.join.path("/data/guyf/1/diff/examples/controlnet/data",a)
# print(patth)
control_image = load_image("/data/guyf/1/diff/examples/controlnet/data/image_test_part003_00000001_source.png")
prompt = "demoire"

# generate image
generator = torch.manual_seed(0)
image = pipe(
    prompt, num_inference_steps=20, generator=generator, image=control_image
).images[0]
image.save("./output_4-26.png")