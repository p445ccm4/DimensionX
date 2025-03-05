import torch
from diffusers import FluxPipeline

class Flux():
    def __init__(self):
        self.pipe = FluxPipeline.from_pretrained(
            "./models/FLUX.1-dev", 
            torch_dtype=torch.bfloat16
        ).to("cuda")

    def gen_image(self, prompt):
        image = self.pipe(
            prompt=prompt,
            negative_prompt="blurry, low quality, human, people, logo, watermark",
            num_inference_steps=50, 
            guidance_scale=30.0,
            width=1280, 
            height=720,
        ).images[0]

        return image