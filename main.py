from IPython.display import Audio
import os
import re
import glob
import json
import tempfile
import math
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
import numpy as np
import commons
import utils
import argparse
import subprocess
from data_utils import TextAudioLoader, TextAudioCollate, TextAudioSpeakerLoader, TextAudioSpeakerCollate
from models import SynthesizerTrn
from scipy.io.wavfile import write
import torchaudio
from chatterbox.vc import ChatterboxVC
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from scipy.io.wavfile import write as wav_write
import soundfile as sf
import ngrok
from pydub import AudioSegment

ngrok.set_auth_token("2yLs2GMfIDMVFjRPbszkcsBOHR4_7cy3wdQirrMzmhLyi1emg")

# Automatically detect the best available device
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


    # -------- FastAPI setup --------
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST"],
    allow_headers=["*"],
)

print(f"Using device: {device}")
model2 = ChatterboxVC.from_pretrained(device)

TARGET_VOICE_PATH = "D:\\Rippleberry\\ttsbackend\\test.wav"
save_path = "D:\\Rippleberry\\ttsbackend\\output.wav"
final_save_path = "D:\\Rippleberry\\ttsbackend\\finalout.wav"
ckpt_dir = "./urd-script_arabic"
LANG = "urd-script_arabic"

def preprocess_char(text, lang=None):
    """
    Special treatement of characters in certain languages
    """
    print(lang)
    if lang == 'ron':
        text = text.replace("ț", "ţ")
    return text

class TextMapper(object):
    def __init__(self, vocab_file):
        self.symbols = [x.replace("\n", "") for x in open(vocab_file, encoding="utf-8").readlines()]
        self.SPACE_ID = self.symbols.index(" ")
        self._symbol_to_id = {s: i for i, s in enumerate(self.symbols)}
        self._id_to_symbol = {i: s for i, s in enumerate(self.symbols)}

    def text_to_sequence(self, text, cleaner_names):
        '''Converts a string of text to a sequence of IDs corresponding to the symbols in the text.
        Args:
        text: string to convert to a sequence
        cleaner_names: names of the cleaner functions to run the text through
        Returns:
        List of integers corresponding to the symbols in the text
        '''
        sequence = []
        clean_text = text.strip()
        for symbol in clean_text:
            symbol_id = self._symbol_to_id[symbol]
            sequence += [symbol_id]
        return sequence

    def uromanize(self, text, uroman_pl):
        iso = "xxx"
        with tempfile.NamedTemporaryFile() as tf, \
             tempfile.NamedTemporaryFile() as tf2:
            with open(tf.name, "w") as f:
                f.write("\n".join([text]))
            cmd = f"perl " + uroman_pl
            cmd += f" -l {iso} "
            cmd +=  f" < {tf.name} > {tf2.name}"
            os.system(cmd)
            outtexts = []
            with open(tf2.name) as f:
                for line in f:
                    line =  re.sub(r"\s+", " ", line).strip()
                    outtexts.append(line)
            outtext = outtexts[0]
        return outtext

    def get_text(self, text, hps):
        text_norm = self.text_to_sequence(text, hps.data.text_cleaners)
        if hps.data.add_blank:
            text_norm = commons.intersperse(text_norm, 0)
        text_norm = torch.LongTensor(text_norm)
        return text_norm

    def filter_oov(self, text):
        val_chars = self._symbol_to_id
        txt_filt = "".join(list(filter(lambda x: x in val_chars, text)))
        print(f"text after filtering OOV: {txt_filt}")
        return txt_filt

def preprocess_text(txt, text_mapper, hps, uroman_dir=None, lang=None):
    txt = preprocess_char(txt, lang=lang)
    is_uroman = hps.data.training_files.split('.')[-1] == 'uroman'
    if is_uroman:
        with tempfile.TemporaryDirectory() as tmp_dir:
            if uroman_dir is None:
                cmd = f"git clone git@github.com:isi-nlp/uroman.git {tmp_dir}"
                print(cmd)
                subprocess.check_output(cmd, shell=True)
                uroman_dir = tmp_dir
            uroman_pl = os.path.join(uroman_dir, "bin", "uroman.pl")
            print(f"uromanize")
            txt = text_mapper.uromanize(txt, uroman_pl)
            print(f"uroman text: {txt}")
    txt = txt.lower()
    txt = text_mapper.filter_oov(txt)
    return txt



print(f"Run inference with {device}")
vocab_file = f"{ckpt_dir}/vocab.txt"
config_file = f"{ckpt_dir}/config.json"
assert os.path.isfile(config_file), f"{config_file} doesn't exist"
hps = utils.get_hparams_from_file(config_file)
text_mapper = TextMapper(vocab_file)


net_g = SynthesizerTrn(
    len(text_mapper.symbols),
    hps.data.filter_length // 2 + 1,
    hps.train.segment_size // hps.data.hop_length,
    **hps.model)
net_g.to(device)
_ = net_g.eval()

g_pth = f"{ckpt_dir}/G_100000.pth"
print(f"load {g_pth}")

_ = utils.load_checkpoint(g_pth, net_g, None)

#txt = "ایران کا اسرائیل پر اب تک کا سب سے بڑا حملہ، اسرائیلی میڈیا کا بڑی تباہی کا اعتراف، متعدد زخمی"


#print(f"text: {txt}")
@app.post("/generate-audio")
async def generate_audio(
    text: str = Form(...),
    target_voice: UploadFile = File(...)
):
    # 1) Text preprocessing & model1 inference (unchanged)
    text = preprocess_text(text, text_mapper, hps, lang=LANG)
    stn_tst = text_mapper.get_text(text, hps)
    with torch.no_grad():
        x_tst = stn_tst.unsqueeze(0).to(device)
        x_tst_lengths = torch.LongTensor([stn_tst.size(0)]).to(device)
        hyp = net_g.infer(
            x_tst, x_tst_lengths,
            noise_scale=0.81, noise_scale_w=0.5, length_scale=0.85
        )[0][0,0].cpu().float().numpy()

    # 2) Save model1 output to a temp WAV
    save_fd, save_path = tempfile.mkstemp(suffix=".wav")
    os.close(save_fd)
    waveform = torch.tensor(hyp).unsqueeze(0)  # [1, T]
    torchaudio.save(save_path, waveform, sample_rate=hps.data.sampling_rate)

    # 3) Write the uploaded file out, detect its extension
    orig_suffix = os.path.splitext(target_voice.filename)[1].lower() or ".wav"
    orig_fd, orig_path = tempfile.mkstemp(suffix=orig_suffix)
    os.close(orig_fd)
    with open(orig_path, "wb") as f:
        f.write(await target_voice.read())

    # 4) If it’s not already WAV, convert to WAV
    if orig_suffix in (".mp4", ".m4a", ".mp3"):
        wav_fd, wav_path = tempfile.mkstemp(suffix=".wav")
        os.close(wav_fd)
        AudioSegment.from_file(orig_path).export(wav_path, format="wav")
        os.remove(orig_path)      # drop the original
        target_wav_path = wav_path
    else:
        target_wav_path = orig_path

    # 5) Run your secondary model
    wav = model2.generate(
        audio=save_path,
        target_voice_path=target_wav_path,
    )
    final_fd, final_save_path = tempfile.mkstemp(suffix=".wav")
    os.close(final_fd)
    torchaudio.save(final_save_path, wav, model2.sr)

    # 6) Clean up temp files we no longer need
    os.remove(save_path)
    os.remove(target_wav_path)

    # 7) Stream back the result
    def iterfile():
        with open(final_save_path, "rb") as file_like:
            yield from file_like

    return StreamingResponse(iterfile(), media_type="audio/wav")

def main():
    import uvicorn
    uvicorn.run(
        "main:app",       # "module:attribute" where your FastAPI() instance lives
        host="0.0.0.0",
        port=8001,  # Enable auto-reload for development
    )

if __name__ == "__main__":
    main()