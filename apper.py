from fastapi import FastAPI, Form, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from transformers import pipeline
import torch
import tempfile
import os
import torchaudio
import numpy as np
from chatterbox.vc import ChatterboxVC
from scipy.io.wavfile import write as wav_write
import soundfile as sf
import noisereduce as nr

# Initialize FastAPI app
app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Update to your frontend domain in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Device setup
device = "cuda:0" if torch.cuda.is_available() else "cpu"

# Load models once
pipe = pipeline("text-to-audio", model="facebook/mms-tts-urd-script_arabic", device=device)
vc_model = ChatterboxVC.from_pretrained(device)

# Output file path
FINAL_OUTPUT_PATH = "finalouter.wav"

@app.post("/generate-audio/")
async def generate_audio(
    text: str = Form(...),
    target_voice: UploadFile = File(...)
):
    # Step 1: TTS pipeline
    outputs = pipe(text)
    audio_array = outputs["audio"]
    audio_array = np.squeeze(audio_array, axis=0)
    sampling_rate = outputs["sampling_rate"]

    # 3) Save uploaded target-voice to temp file
    tgt_fd, tgt_path = tempfile.mkstemp(
        suffix=os.path.splitext(target_voice.filename)[1] or ".wav"
    )

    os.close(tgt_fd)
    with open(tgt_path, "wb") as f:
        f.write(await target_voice.read())

    # 3 Save generated audio to a temporary WAV file
    tts_fd, tts_path = tempfile.mkstemp(suffix=".wav")
    os.close(tts_fd)
    wav_write(tts_path, sampling_rate, audio_array)

    #denoise(tgt_path, outpath)
    # Step 3: Voice conversion using file path
    vc_wav = vc_model.generate(audio=tts_path, target_voice_path=tgt_path)

    # Step 4: Save final output
    torchaudio.save(FINAL_OUTPUT_PATH, vc_wav, vc_model.sr)

    # Step 5: Stream back result
    def iterfile():
        with open(FINAL_OUTPUT_PATH, "rb") as file_like:
            yield from file_like


    return StreamingResponse(iterfile(), media_type="audio/wav")

def main():
    import uvicorn
    uvicorn.run(
        "apper:app",  # "module:attribute" where your FastAPI() instance lives
        host="0.0.0.0",
        port=8001,  # Enable auto-reload for development
    )

if __name__ == "__main__":
    main()