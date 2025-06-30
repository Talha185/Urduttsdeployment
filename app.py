import io
import os
import tempfile
import torch
import utils
import commons
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from scipy.io.wavfile import write as wav_write
import soundfile as sf
from models import SynthesizerTrn
from chatterbox.vc import ChatterboxVC
import torchaudio
import soundfile as sf
import noisereduce as nr

# -------- Device & model setup (runs once at startup) --------
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print(f"Using device: {device}")

# Load VC model
vc_model = ChatterboxVC.from_pretrained(device)
final_save_path = "D:\\Rippleberry\\ttsbackend\\finaloutcustomdenoised.wav"
outpath = "D:\\Rippleberry\\ttsbackend\\out1.wav"


# Paths for your trained TTS checkpoint
CKPT_DIR = "./urd-script_arabic"
VOCAB_FILE = os.path.join(CKPT_DIR, "vocab.txt")
CONFIG_FILE = os.path.join(CKPT_DIR, "config.json")
G_PTH       = os.path.join(CKPT_DIR, "G_100000.pth")

# Load hyperparams & text mapper
hps = utils.get_hparams_from_file(CONFIG_FILE)

class TextMapper:
    def __init__(self, vocab_file):
        symbols = [x.strip() for x in open(vocab_file, encoding="utf-8")]
        self.symbols = symbols
        self._sym2id = {s:i for i,s in enumerate(symbols)}
    def text_to_sequence(self, text, _):
        return [self._sym2id[s] for s in text if s in self._sym2id]
    def get_text(self, txt, hps):
        seq = self.text_to_sequence(txt, hps.data.text_cleaners)
        if hps.data.add_blank:
            seq = commons.intersperse(seq, 0)
        return torch.LongTensor(seq)

def preprocess_char(text, lang=None):
    if lang == 'ron':
        text = text.replace("ț", "ţ")
    return text

def preprocess_text(txt, mapper, hps, lang=None):
    txt = preprocess_char(txt, lang=lang)
    txt = txt.lower()
    # filter OOV
    txt = "".join([c for c in txt if c in mapper._sym2id])
    return txt

text_mapper = TextMapper(VOCAB_FILE)

# Build & load TTS generator
net_g = SynthesizerTrn(
    len(text_mapper.symbols),
    hps.data.filter_length // 2 + 1,
    hps.train.segment_size // hps.data.hop_length,
    **hps.model
).to(device).eval()

utils.load_checkpoint(G_PTH, net_g, None)
print("TTS model loaded.")

# -------- FastAPI setup --------
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST"],
    allow_headers=["*"],
)

def denoise(audio_path, output_path):
    # Load audio using SoundFile if possible for better handling of file formats
    data, sr = sf.read(audio_path)
    
    # Reduce noise
    reduced_noise = nr.reduce_noise(y=data, sr=sr)
    sf.write(output_path, reduced_noise, sr)
    print("Denoising successful")

@app.post("/generate-audio")
async def generate_audio(
    text: str = Form(...),
    target_voice: UploadFile = File(...)
):
    # 1) TTS: preprocess and infer
    print("Generating audio for text:", text)
    clean = preprocess_text(text, text_mapper, hps, lang="urd-script_arabic")
    seq = text_mapper.get_text(clean, hps).unsqueeze(0).to(device)
    lengths = torch.LongTensor([seq.size(1)]).to(device)

    print("infering TTS...")
    with torch.no_grad():
        audio_np = net_g.infer(
            seq, lengths,
            noise_scale=0.669,
            noise_scale_w=1.0,
            length_scale=0.92
        )[0][0,0].cpu().numpy()

    print("TTS inference complete.")

    print(f"Audio array shape: {audio_np.shape}")
    print(f"Audio array dtype: {audio_np.dtype}")
    print(type(audio_np))
    print(audio_np)
    # 2) Write TTS output to temp WAV
    tts_fd, tts_path = tempfile.mkstemp(suffix=".wav")
    os.close(tts_fd)
    wav_write(tts_path, hps.data.sampling_rate, audio_np)

    # 3) Save uploaded target-voice to temp file
    tgt_fd, tgt_path = tempfile.mkstemp(
        suffix=os.path.splitext(target_voice.filename)[1] or ".wav"
    )
    os.close(tgt_fd)
    with open(tgt_path, "wb") as f:
        f.write(await target_voice.read())

    print(f"Target voice saved to {tgt_path}")
    # 4) Voice-conversion
    denoise(tgt_path, outpath)
    try:
        vc_wav = vc_model.generate(
            audio=tts_path,
            target_voice_path=tgt_path
        )
        torchaudio.save(final_save_path, vc_wav, vc_model.sr)
    except Exception as e:
        raise HTTPException(status_code=500, detail="Voice conversion failed")
    finally:
        # clean up temps
        os.remove(tts_path)
        os.remove(tgt_path)

    # 5) Stream back as WAV
    buffer = io.BytesIO()
    # vc_wav is a numpy array or torch.Tensor [T], convert if needed:
    data = vc_wav.cpu().numpy() if hasattr(vc_wav, "cpu") else vc_wav
    if vc_wav is None:
        raise HTTPException(status_code=500, detail="Voice conversion failed; got None")

    #if len(data.shape) != 1:
        #raise HTTPException(status_code=500, detail=f"Invalid audio shape: {data.shape}")
    print("Writing to buffer", type(data), data.shape)
    sf.write(buffer, data.squeeze(), vc_model.sr, format="WAV")
    print("Audio written to buffer")
    buffer.seek(0)
    return StreamingResponse(buffer, media_type="audio/wav")

def main():
    import uvicorn
    uvicorn.run(
        "app:app",       # "module:attribute" where your FastAPI() instance lives
        host="0.0.0.0",
        port=8000,
        reload=True,
    )

if __name__ == "__main__":
    main()