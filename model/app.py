import os
import subprocess
import torchaudio
import torch
from PIL import Image
from flask import Flask, request, jsonify

app = Flask(__name__)