import torch
from diffusers import StableDiffusionPipeline

pipe = StableDiffusionPipeline.from_pretrained("bk-sdm-v2-small", torch_dtype=torch.float16)
pipe = pipe.to("cuda")

prompt = "stingray"
image = pipe(prompt).images[0]

image.save("example.png")