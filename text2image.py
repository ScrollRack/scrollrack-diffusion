import torch
from torch import autocast
from diffusers import StableDiffusionPipeline, LMSDiscreteScheduler
import upsampler
import uploader
import time
import requests

torch.backends.cuda.matmul.allow_tf32 = True

device = 'cuda' if torch.cuda.is_available() else 'cpu'

negative_prompt = "lowres, text, error, cropped, worst quality, low quality, jpeg artifacts, ugly, duplicate, morbid, mutilated, out of frame, extra fingers, mutated hands, poorly drawn hands, poorly drawn face, mutation, deformed, blurry, dehydrated, bad anatomy, bad proportions, extra limbs, cloned face, disfigured, gross proportions, malformed limbs, missing arms, missing legs, extra arms, extra legs, fused fingers, too many fingers, long neck"

pipe = StableDiffusionPipeline.from_pretrained(
    "./stable-diffusion-v1-5",
).to(device)

def dummy(images, **kwargs):
    return images, False

pipe.safety_checker = dummy

def create_latents (width, height, seed):
    generator = torch.Generator(device=device)
    new_seed = generator.seed() if seed == 0 or seed is None else seed
    generator = generator.manual_seed(new_seed)

    image_latents = torch.randn(
        (1, pipe.unet.in_channels, height // 8,  width // 8),
        device=device,
        generator=generator
    )

    return new_seed, image_latents

def generate_image(prompt, seed, width, height, steps, iterations, guidance_scale):
    start_time = time.time()
    with autocast("cuda"):
        image_seed, latents = create_latents(width, height, seed)
        image = pipe(prompt, negative_prompt=negative_prompt, guidance_scale=guidance_scale, width=width, height=height, num_inference_steps=steps, latents=latents).images[0]

        image = upsampler.upscale(image)
        file_url = uploader.send(image)

    return file_url, time.time() - start_time, image_seed

def process(prompt, args):
    """Create image and upload to S3"""
    webhook = args.get('webhook_url')

    if not webhook:
        return
        
    try:
        image, time, image_seed = generate_image(
            prompt = prompt,
            seed = args.get('seed', 0),
            width = args.get('width', 512),
            height = args.get('height', 512),
            steps = args.get('steps', 40),
            iterations = args.get('iterations', 1),
            guidance_scale = args.get('scale', 7.5),
        )

        t = requests.post(webhook, json={
            "images": [{ "seed": image_seed, "image": image }], "time": time, "job_id": args.get('job_id')
        })

        pass
    except Exception as e:
        print(e)