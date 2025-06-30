from transformers import pipeline
import torch
from IPython.display import Audio
import scipy.io.wavfile as wavfile
import numpy as np
from chatterbox.vc import ChatterboxVC
import torchaudio
import os
import tempfile




device = "cuda:0" if torch.cuda.is_available() else "cpu"

pipe = pipeline("text-to-audio", model="facebook/mms-tts-urd-script_arabic", device=device)
vc_model = ChatterboxVC.from_pretrained(device)

final_save_path = "D:\\Rippleberry\\ttsbackend\\finalout.wav"

outputs = pipe("دوسری جانب اسرائیل نے جو اعداد و شمار دنیا کے سامنے رکھے اس کے مطابق ایرانی حملوں کے نتیجے میں درجن سے زائد اسرائیلی ہلاک ہوئے تاہم اس جنگ میں دونوں ملکوں کو روزانہ سیکڑوں ملین ڈالر کا نقصان بھی برداشت کرنا پڑا۔")
print(outputs)
#Audio(outputs["audio"], rate=outputs["sampling_rate"])
sampling_rate = outputs["sampling_rate"]
print(f"Sampling Rate: {sampling_rate}")
audio_array = outputs["audio"]
#audio_array = outputs["audio"].squeeze()
#audio_array /=1.414
#audio_array *= 32767
#audio_array = audio_array.astype(np.int16)
#wavfile.write("output.wav", sampling_rate, audio_array)
tgt_fd, tgt_path = tempfile.mkstemp(
        suffix=os.path.splitext(target_voice.filename)[1] or ".wav"
    )

vc_wav = vc_model.generate(
            audio=audio_array,
            target_voice_path=tgt_path
        )
torchaudio.save(final_save_path, vc_wav, vc_model.sr)
