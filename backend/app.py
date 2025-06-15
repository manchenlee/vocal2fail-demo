# backend/app.py
from flask import Flask, request, send_file, render_template
import tempfile
import torch
import json
from io import BytesIO
from inference import audio_infer
from vocoder import mel2audio
from scipy.io.wavfile import write
from env import AttrDict
import os

app = Flask(
    __name__,
    template_folder=os.path.join(os.path.dirname(__file__), "../templates"),
    static_folder=os.path.join(os.path.dirname(__file__), "../static")
)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/process", methods=["POST"])
def process_audio():
    level = request.form.get("level")
    audio_file = request.files["audio"]
    """
    if level == 'default':
        multi = 0
    elif level == 'bad':
        multi = 5
    else:
        multi = 10
    """
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_input:
        audio_file.save(tmp_input.name)
        tmp_input_path = tmp_input.name

    print('inference... (VAE-GAN audio2mel)')
    converted = audio_infer(tmp_input_path, int(level))
    print('complete.')

    config_file = 'config.json'
    with open(config_file) as f:
        data = f.read()

    global h
    json_config = json.loads(data)
    h = AttrDict(json_config)

    torch.manual_seed(h.seed)
    print('inference... (BIGVSAN mel2audio)')
    converted_audio = mel2audio(converted, h)
    print('complete.')

    output_io = BytesIO()
    write(output_io, h.sampling_rate, converted_audio)
    output_io.seek(0)

    return send_file(output_io, mimetype="audio/wav", as_attachment=True, download_name="output.wav")


#if __name__ == "__main__":
#    app.run(debug=True)
