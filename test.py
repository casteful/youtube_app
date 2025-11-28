import os
import torch
from TTS.api import TTS

print(torch.cuda.is_available())
print(torch.version.cuda)
print(torch.cuda.device_count())

MODEL = "tts_models/en/vctk/vits"
TEXT = "This is a voice preview for speaker verification."
OUTPUT_DIR = "previews"

USE_GPU = torch.cuda.is_available()
print(USE_GPU)
tts = TTS(model_name=MODEL, gpu=USE_GPU, progress_bar=False)

os.makedirs(OUTPUT_DIR, exist_ok=True)

tts = TTS(model_name=MODEL)

def sanitize_filename(name):
    return "".join(c if c.isalnum() or c in " -_." else "_" for c in name)

#safe_name = sanitize_filename("test")
#out_file = os.path.join(OUTPUT_DIR, f"{safe_name}.wav")
#tts.tts_to_file(text=TEXT, file_path=out_file)

for spk in tts.speakers:
    safe_name = sanitize_filename(spk)
    out_file = os.path.join(OUTPUT_DIR, f"{safe_name}.wav")
    tts.tts_to_file(text=TEXT, file_path=out_file, speaker=spk)
    print("Generated:", out_file)

