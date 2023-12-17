import os
import subprocess
import torchaudio
import torch
from PIL import Image
from flask import Flask, request, jsonify

app = Flask(__name__)

def pad_image(image):
    w, h = image.size
    if w == h:
        return image
    elif w > h:
        new_image = Image.new(image.mode, (w, w), (0, 0, 0))
        new_image.paste(image, (0, (w - h) // 2))
        return new_image
    else:
        new_image = Image.new(image.mode, (h, h), (0, 0, 0))
        new_image.paste(image, ((h - w) // 2, 0))
        return new_image
    
def calculate(image_in, audio_in):
    waveform, sample_rate = torchaudio.load(audio_in)
    waveform = torch.mean(waveform, dim=0, keepdim=True)
    torchaudio.save("/home/demo/source/audio.wav", waveform, sample_rate, encoding="PCM_S", bits_per_sample=16)
    image = Image.open(image_in)
    image = pad_image(image)
    image.save("/home/demo/source/image.png")

