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
from moviepy.editor import ImageClip, CompositeVideoClip, AudioFileClip

# ================= APPEARANCE =================
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("dark-blue")

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
app.title("Horror TTS + Video Studio")
app.geometry("900x760")

# ================= MODEL =================
ctk.CTkLabel(app, text="TTS Model").pack(pady=(8,0))
model_entry = ctk.CTkEntry(app, width=520)
model_entry.insert(0, "tts_models/en/vctk/vits")
model_entry.pack()

# ================= SPEAKERS =================
ctk.CTkLabel(app, text="Narrator Speaker").pack(pady=(8,0))
narrator_entry = ctk.CTkEntry(app, width=200)
narrator_entry.insert(0, "p226")
narrator_entry.pack()

ctk.CTkLabel(app, text="Dialogue Speaker").pack(pady=(8,0))
dialogue_entry = ctk.CTkEntry(app, width=200)
dialogue_entry.insert(0, "p229")
dialogue_entry.pack()

# ================= SPEED =================
speed_frame = ctk.CTkFrame(app)
speed_frame.pack(pady=10, fill="x", padx=30)

speed_calm = ctk.CTkSlider(speed_frame, from_=0.7, to=1.4)
speed_tense = ctk.CTkSlider(speed_frame, from_=0.7, to=1.4)
speed_sad = ctk.CTkSlider(speed_frame, from_=0.7, to=1.4)

speed_calm.set(1.0)
speed_tense.set(0.98)
speed_sad.set(0.99)

ctk.CTkLabel(speed_frame, text="Speed Calm").grid(row=0, column=0, sticky="w")
speed_calm.grid(row=0, column=1, sticky="ew")
ctk.CTkLabel(speed_frame, text="Speed Tense").grid(row=1, column=0, sticky="w")
speed_tense.grid(row=1, column=1, sticky="ew")
ctk.CTkLabel(speed_frame, text="Speed Sad").grid(row=2, column=0, sticky="w")
speed_sad.grid(row=2, column=1, sticky="ew")

speed_frame.columnconfigure(1, weight=1)

# ================= GPU / CPU TOGGLE =================
device_toggle = ctk.StringVar(value="GPU" if torch.cuda.is_available() else "CPU")
ctk.CTkLabel(app, text="Device").pack()
ctk.CTkOptionMenu(app, values=["GPU", "CPU"], variable=device_toggle).pack()

# ================= FILES =================
story_path = ctk.StringVar()
output_path = ctk.StringVar(value="output.wav")
ambience_path = ctk.StringVar()
images_path = ctk.StringVar()

def browse_story():
    story_path.set(filedialog.askopenfilename(filetypes=[("Text","*.txt")]))
def browse_output():
    output_path.set(filedialog.asksaveasfilename(defaultextension=".wav"))
def browse_ambience():
    ambience_path.set(filedialog.askopenfilename(filetypes=[("Audio","*.wav")]))
def browse_images():
    images_path.set(filedialog.askdirectory())

file_frame = ctk.CTkFrame(app)
file_frame.pack(pady=12)

ctk.CTkButton(file_frame, text="Story File", command=browse_story).grid(row=0, column=0, padx=6)
ctk.CTkEntry(file_frame, textvariable=story_path, width=420).grid(row=0, column=1)

ctk.CTkButton(file_frame, text="Output WAV", command=browse_output).grid(row=1, column=0, padx=6)
ctk.CTkEntry(file_frame, textvariable=output_path, width=420).grid(row=1, column=1)

ctk.CTkButton(file_frame, text="Ambience WAV", command=browse_ambience).grid(row=2, column=0, padx=6)
ctk.CTkEntry(file_frame, textvariable=ambience_path, width=420).grid(row=2, column=1)

ctk.CTkButton(file_frame, text="Images Folder", command=browse_images).grid(row=3, column=0, padx=6)
ctk.CTkEntry(file_frame, textvariable=images_path, width=420).grid(row=3, column=1)

# ================= AMBIENCE VOLUME =================
ctk.CTkLabel(app, text="Ambience Volume").pack()
ambience_vol = ctk.CTkSlider(app, from_=0, to=40)
ambience_vol.set(20)
ambience_vol.pack(fill="x", padx=60)

# ================= VIDEO SETTINGS =================
video_frame = ctk.CTkFrame(app)
video_frame.pack(pady=12)

ctk.CTkLabel(video_frame, text="Fade (s)").grid(row=0, column=0)
fade_var = ctk.DoubleVar(value=1.0)
ctk.CTkEntry(video_frame, textvariable=fade_var, width=70).grid(row=0, column=1)

ctk.CTkLabel(video_frame, text="Zoom").grid(row=0, column=2)
zoom_var = ctk.DoubleVar(value=1.05)
ctk.CTkEntry(video_frame, textvariable=zoom_var, width=70).grid(row=0, column=3)

ctk.CTkLabel(video_frame, text="FPS").grid(row=0, column=4)
fps_var = ctk.IntVar(value=24)
ctk.CTkEntry(video_frame, textvariable=fps_var, width=70).grid(row=0, column=5)

# ================= LOG =================
log_box = ctk.CTkTextbox(app, width=860, height=120)
log_box.pack(pady=10)

def log(msg):
    log_box.insert("end", msg + "\n")
    log_box.see("end")

# ================= CORE RUNNER =================
def run_pipeline():
    model = model_entry.get().strip()
    narrator = narrator_entry.get().strip()
    dialogue = dialogue_entry.get().strip()
    story = story_path.get()
    output = output_path.get()
    ambience = ambience_path.get()
    images = images_path.get()

    use_gpu = device_toggle.get() == "GPU"
    log(f"Device: {'CUDA' if use_gpu else 'CPU'}")

    log("Loading model...")
    tts = TTS(model_name=model, gpu=use_gpu, progress_bar=False)

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

    # ==== Ambience Mix ====
    if os.path.exists(ambience):
        amb = AudioSegment.from_file(ambience).low_pass_filter(400)
        amb = amb - ambience_vol.get()
        amb_loop = amb * (len(final_audio) // len(amb) + 1)
        final_audio = final_audio.overlay(amb_loop[:len(final_audio)])

    final_audio = effects.normalize(final_audio)
    final_audio.export(output, format="wav")
    log("✅ Audio synthesized.")

    # ==== VIDEO ====
    if os.path.isdir(images):
        clips = []
        images_list = sorted([os.path.join(images, f)
                               for f in os.listdir(images)
                               if f.lower().endswith((".png",".jpg",".jpeg"))])

        duration = final_audio.duration_seconds / max(len(images_list),1)
        t = 0

        for img in images_list:
            clip = ImageClip(img).set_duration(duration)
            clip = clip.fadein(fade_var.get()).fadeout(fade_var.get())
            clip = clip.resize(lambda x: 1 + (zoom_var.get()-1)*(x/duration))
            clip = clip.set_start(t)
            clips.append(clip)
            t += duration

        video = CompositeVideoClip(clips)
        audio_clip = AudioFileClip(output)
        video = video.set_audio(audio_clip)

        out_video = os.path.splitext(output)[0] + ".mp4"
        video.write_videofile(out_video, fps=fps_var.get(),
                              codec="libx264", audio_codec="aac")
        log("✅ Video created: " + out_video)

    log("\n✅ FULL PIPELINE COMPLETE")

def run_thread():
    threading.Thread(target=run_pipeline, daemon=True).start()

# ================= RUN BUTTON =================
ctk.CTkButton(app, text="▶ RUN FULL PIPELINE", height=45,
              fg_color="#8b0000", command=run_thread).pack(pady=16)

app.mainloop()