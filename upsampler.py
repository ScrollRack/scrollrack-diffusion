import cv2
import glob
import sys
import os
import requests
import time
import numpy as np
from os.path import join, dirname
from dotenv import load_dotenv
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan.archs.srvgg_arch import SRVGGNetCompact
from PIL import Image
from realesrgan import RealESRGANer

dotenv_path = join(dirname(__file__), '.env')
load_dotenv(dotenv_path)

model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64,
                num_block=23, num_grow_ch=32, scale=4)
model_path = os.path.join('./' + os.environ.get('MODEL_PATH'))


def upscale( image, factor=2 ):
    upsampler = RealESRGANer(
        scale = 4,
        model_path = model_path,
        model = model,
        tile = 0,
        tile_pad = 10,
        pre_pad = 0,
        gpu_id = 0
    )

    tile = 0
    tile_pad = 10
    pre_pad = 0
    fp32 = True
    gpu_id = None
    netscale = 4

    data = np.array(image)[:, :, ::-1].copy()
    output, _ = upsampler.enhance(data, outscale=factor)
    image = Image.fromarray(output[:, :, ::-1].copy())

    return image