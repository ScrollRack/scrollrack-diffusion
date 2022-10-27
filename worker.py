import os, sys
import redis
import time
import json
import text2image
import requests
from dotenv import load_dotenv
from os.path import join, dirname
from dotenv import load_dotenv

dotenv_path = join(dirname(__file__), '.env')
load_dotenv(dotenv_path)

r = redis.Redis(
    host= os.environ.get('REDIS_HOST', 'localhost'),
    port= os.environ.get('REDIS_PORT', '6379'),
    password= os.environ.get('REDIS_PASSWORD'),
    ssl=False
)

while True:
    try:
        t2i_payload = r.lpop('generate_images')
        upscale_payload = r.lpop('upscale_images')

        if t2i_payload:
            data = json.loads(t2i_payload)
            text2image.process(data['prompt'], data)
    except:
        pass

    # if upscale_payload:
        # data = json.loads(upscale_payload)
        # upscale_image(data)
    
    time.sleep(0.25)