import requests
import upsampler
import uploader
from PIL import Image

def upscale(payload):
    image_url = payload.get('image_url')
    factor = payload.get('scale', 2)
    webhook = payload.get('webhook_url')
    job_id = payload.get('job_id')

    if job_id and webhook and image_url:

        image_data = requests.get(image_url, stream=True).raw
        image = Image.open(image_data)
        image = upsampler.upscale(image, factor=factor)
        file_url = uploader.send(image)

        if file_url:
            requests.post(webhook, json={ 'file_url': file_url, 'job_id': job_id })
