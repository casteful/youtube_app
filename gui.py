import customtkinter as ctk
from tkinter import filedialog
import torch
from TTS.api import TTS
import numpy as np
import soundfile as sf
from pydub import AudioSegment, effects
import librosa
import io
import re
import threading
import os

# ================= APPEARANCE =================
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("dark-blue")

USE_GPU = torch.cuda.is_available()

# ================= EMOTION =================
def detect_emotion(text):
    t = text.lower()
    if re.search(r"(scream|blood|dark|shadow|knife|dead|fear|chase)", t):
        return "tense"
    if re.search(r"(cry|alone|sad|tears|empty)", t):
        return "sad"
    return "calm"

# ================= APP =================
app = ctk.CTk()
app.title("Modern Local TTS Studio")
app.geometry("820x640")

# ================= INPUTS =================
ctk.CTkLabel(app, text="TTS Model").pack(pady=(10,0))
model_entry = ctk.CTkEntry(app, width=520)
model_entry.insert(0, "tts_models/en/vctk/vits")
model_entry.pack()

ctk.CTkLabel(app, text="Narrator Speaker").pack(pady=(10,0))
narrator_entry = ctk.CTkEntry(app, width=200)
narrator_entry.insert(0, "p226")
narrator_entry.pack()

ctk.CTkLabel(app, text="Dialogue Speaker").pack(pady=(10,0))
dialogue_entry = ctk.CTkEntry(app, width=200)
dialogue_entry.insert(0, "p229")
dialogue_entry.pack()

# ================= SPEED =================
ctk.CTkLabel(app, text="Speed (Calm)").pack()
speed_calm = ctk.CTkSlider(app, from_=0.7, to=1.4)
speed_calm.set(1.0)
speed_calm.pack(fill="x", padx=40)

ctk.CTkLabel(app, text="Speed (Tense)").pack()
speed_tense = ctk.CTkSlider(app, from_=0.7, to=1.4)
speed_tense.set(0.98)
speed_tense.pack(fill="x", padx=40)

ctk.CTkLabel(app, text="Speed (Sad)").pack()
speed_sad = ctk.CTkSlider(app, from_=0.7, to=1.4)
speed_sad.set(0.99)
speed_sad.pack(fill="x", padx=40)

# ================= FILES =================
story_path = ctk.StringVar()
output_path = ctk.StringVar(value="output.wav")

def browse_story():
    story_path.set(filedialog.askopenfilename(filetypes=[("Text","*.txt")]))

def browse_output():
    output_path.set(filedialog.asksaveasfilename(defaultextension=".wav"))

file_frame = ctk.CTkFrame(app)
file_frame.pack(pady=12)

ctk.CTkButton(file_frame, text="Story File", command=browse_story).grid(row=0, column=0, padx=10)
ctk.CTkEntry(file_frame, textvariable=story_path, width=420).grid(row=0, column=1)

ctk.CTkButton(file_frame, text="Output WAV", command=browse_output).grid(row=1, column=0, padx=10)
ctk.CTkEntry(file_frame, textvariable=output_path, width=420).grid(row=1, column=1)

# ================= LOG =================
log_box = ctk.CTkTextbox(app, width=760, height=200)
log_box.pack(pady=8)

def log(msg):
    log_box.insert("end", msg + "\n")
    log_box.see("end")

# ================= CORE RUNNER =================
def run_tts():
    model = model_entry.get().strip()
    narrator = narrator_entry.get().strip()
    dialogue = dialogue_entry.get().strip()
    story = story_path.get()
    output = output_path.get()

    if not os.path.exists(story):
        log("❌ Story file not found.")
        return

    log("Loading model...")
    tts = TTS(model_name=model, gpu=USE_GPU, progress_bar=False)

    with open(story, "r", encoding="utf-8") as f:
        paragraphs = [p.strip() for p in f.read().split("\n\n") if p.strip()]

    final_audio = AudioSegment.silent(0)

    for para in paragraphs:
        sentences = re.split(r'(?<=[.!?])\s+', para)

        for s in sentences:
            if not s.strip():
                continue

            emotion = detect_emotion(s)
            speed = {
                "calm": speed_calm.get(),
                "tense": speed_tense.get(),
                "sad": speed_sad.get()
            }[emotion]

            speaker = dialogue if s.startswith(("\"", "“", "'")) else narrator
            log(f"{emotion.upper()} | {speaker} | {speed:.2f}")

            try:
                wav = tts.tts(text=s, speaker=speaker)
            except:
                wav = tts.tts(text=s)

            wav = np.array(wav, dtype=np.float32)
            wav = librosa.effects.time_stretch(wav, rate=float(speed))

            buf = io.BytesIO()
            sf.write(buf, wav, 24000, format="WAV")
            buf.seek(0)

            seg = AudioSegment.from_file(buf, format="wav")
            final_audio += seg + AudioSegment.silent(250)

    final_audio = effects.normalize(final_audio)
    final_audio.export(output, format="wav")

    log("\n✅ DONE: " + output)

def run_thread():
    threading.Thread(target=run_tts, daemon=True).start()

ctk.CTkButton(app, text="RUN TTS", height=45, command=run_thread).pack(pady=20)

app.mainloop()