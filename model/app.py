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

    pocketsphinx_run = subprocess.run(['pocketsphinx', '-phone_align', 'yes', 'single', '/home/demo/source/audio.wav'], check=True, capture_output=True)
    jq_run = subprocess.run(['jq', '[.w[]|{word: (.t | ascii_upcase | sub("<S>"; "sil") | sub("<SIL>"; "sil") | sub("\\\(2\\\)"; "") | sub("\\\(3\\\)"; "") | sub("\\\(4\\\)"; "") | sub("\\\[SPEECH\\\]"; "SIL") | sub("\\\[NOISE\\\]"; "SIL")), phones: [.w[]|{ph: .t | sub("\\\+SPN\\\+"; "SIL") | sub("\\\+NSN\\\+"; "SIL"), bg: (.b*100)|floor, ed: (.b*100+.d*100)|floor}]}]'], input=pocketsphinx_run.stdout, capture_output=True)
    with open("/home/demo/source/test.json", "w") as f:
        f.write(jq_run.stdout.decode('utf-8').strip())

    os.system(f"cd /home/demo/source/one-shot-talking-face && python3 -B test_script.py --img_path /home/demo/source/image.png --audio_path /home/demo/source/audio.wav --phoneme_path /home/demo/source/test.json --save_dir /home/demo/source/train")
    result_path = "/home/demo/source/train/image_audio.mp4"
    return result_path

@app.route('/api/calculate', methods=['POST'])
def api_calculate():
    if 'image' not in request.files or 'audio' not in request.files:
        return jsonify({'error': 'Image and audio files are required'}), 400

    image_file = request.files['image']
    audio_file = request.files['audio']

    result_path = calculate(image_file, audio_file)
    return jsonify({'result': result_path})

if __name__ == "__main__":
    app.run(debug=True)
