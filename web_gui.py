import gradio as gr
import torch
from TTS.api import TTS
import numpy as np
import soundfile as sf
from pydub import AudioSegment, effects
import io
import re
import os
import tempfile
import zipfile
from moviepy.editor import ImageClip, CompositeVideoClip, AudioFileClip
import librosa

# ----------------- GLOBAL MODEL LOAD -----------------
USE_GPU = torch.cuda.is_available()
TTS_MODEL = "tts_models/en/vctk/vits"
tts_model = TTS(model_name=TTS_MODEL, gpu=USE_GPU, progress_bar=False)

# ----------------- EMOTION DETECTION -----------------
TENSE_RE = r"(scream|blood|dark|shadow|knife|dead|fear|chase)"
SAD_RE = r"(cry|alone|sad|tears|empty)"

def detect_emotion(text):
    t = text.lower()
    if re.search(TENSE_RE, t):
        return "tense"
    if re.search(SAD_RE, t):
        return "sad"
    return "calm"

def get_speed(emotion, calm, tense, sad):
    return {"calm": calm, "tense": tense, "sad": sad}[emotion]

# ----------------- OPTIMIZED TTS PIPELINE -----------------
def tts_pipeline(
    story_text,
    narrator_speaker="p226",
    dialogue_speaker="p229",
    speed_calm=1.0,
    speed_tense=0.98,
    speed_sad=0.99,
    ambience_file=None,
    images_zip=None,
    fade_sec=1.0,
    zoom=1.05,
    fps=24
):
    paragraphs = [p.strip() for p in story_text.split("\n\n") if p.strip()]
    final_audio = AudioSegment.silent(0)

    # ---------- PROCESS PARAGRAPHS ----------
    for para in paragraphs:
        sentences = re.split(r'(?<=[.!?])\s+', para)
        para_audio = AudioSegment.silent(0)
        batch_text = []
        batch_speakers = []

        for sentence in sentences:
            if not sentence.strip():
                continue
            batch_text.append(sentence)
            batch_speakers.append(dialogue_speaker if sentence.startswith(("\"", "“", "'")) else narrator_speaker)

        # Generate TTS for the whole paragraph at once (if model supports batching)
        for txt, spk in zip(batch_text, batch_speakers):
            wav = tts_model.tts(text=txt, speaker=spk)
            wav = np.array(wav, dtype=np.float32)
            emotion = detect_emotion(txt)
            speed = get_speed(emotion, speed_calm, speed_tense, speed_sad)
            wav_stretched = librosa.effects.time_stretch(wav, rate=speed)

            buf = io.BytesIO()
            sf.write(buf, wav_stretched, 24000, format="WAV")
            buf.seek(0)
            seg = AudioSegment.from_file(buf, format="wav")
            para_audio += seg + AudioSegment.silent(250)

        para_audio = para_audio.fade_in(int(fade_sec*1000)).fade_out(int(fade_sec*1000))
        final_audio += para_audio + AudioSegment.silent(500)

    # ---------- AMBIENCE ----------
    if ambience_file is not None:
        amb = AudioSegment.from_file(ambience_file.name).low_pass_filter(400)
        amb = amb - 20
        amb_loop = amb * (len(final_audio) // len(amb) + 1)
        final_audio = final_audio.overlay(amb_loop[:len(final_audio)])

    final_audio = effects.normalize(final_audio)

    # ---------- SAVE AUDIO ----------
    audio_fd, audio_path = tempfile.mkstemp(suffix=".wav")
    final_audio.export(audio_path, format="wav")

    # ---------- VIDEO ----------
    video_path = None
    if images_zip is not None:
        tmp_dir = tempfile.mkdtemp()
        with zipfile.ZipFile(images_zip.name, "r") as zip_ref:
            zip_ref.extractall(tmp_dir)

        images_list = sorted([os.path.join(tmp_dir, f) for f in os.listdir(tmp_dir)
                              if f.lower().endswith((".png",".jpg",".jpeg"))])
        if images_list:
            clips = []
            duration = final_audio.duration_seconds / len(images_list)
            t = 0
            for img in images_list:
                clip = ImageClip(img).set_duration(duration)
                clip = clip.fadein(fade_sec).fadeout(fade_sec)
                clip = clip.resize(lambda x: 1 + (zoom-1)*(x/duration))
                clip = clip.set_start(t)
                clips.append(clip)
                t += duration

            video = CompositeVideoClip(clips)
            video = video.set_audio(AudioFileClip(audio_path))
            video_fd, video_path = tempfile.mkstemp(suffix=".mp4")
            video.write_videofile(video_path, fps=fps, codec="libx264", audio_codec="aac", threads=8)

    return audio_path, video_path

# ----------------- GRADIO UI -----------------
with gr.Blocks() as demo:
    gr.Markdown("## Horror TTS + Video Studio (Ultra-Fast)")

    story_input = gr.Textbox(label="Story Text", lines=12)
    narrator_input = gr.Textbox(label="Narrator Speaker", value="p226")
    dialogue_input = gr.Textbox(label="Dialogue Speaker", value="p229")
    speed_calm = gr.Slider(label="Speed Calm", minimum=0.7, maximum=1.4, value=1.0, step=0.01)
    speed_tense = gr.Slider(label="Speed Tense", minimum=0.7, maximum=1.4, value=0.98, step=0.01)
    speed_sad = gr.Slider(label="Speed Sad", minimum=0.7, maximum=1.4, value=0.99, step=0.01)
    ambience_file = gr.File(label="Ambience WAV (Optional)", file_types=[".wav"])
    images_zip = gr.File(label="Images Folder (Zip)", file_types=[".zip"])
    fade_sec = gr.Slider(label="Fade (s)", minimum=0, maximum=5, value=1.0, step=0.1)
    zoom = gr.Slider(label="Zoom", minimum=1.0, maximum=2.0, value=1.05, step=0.01)
    fps = gr.Slider(label="Video FPS", minimum=1, maximum=60, value=24, step=1)

    run_btn = gr.Button("▶ RUN FULL PIPELINE")
    audio_out = gr.Audio(label="Generated Audio")
    video_out = gr.Video(label="Generated Video (Optional)")

    run_btn.click(
        tts_pipeline,
        inputs=[story_input, narrator_input, dialogue_input,
                speed_calm, speed_tense, speed_sad,
                ambience_file, images_zip, fade_sec, zoom, fps],
        outputs=[audio_out, video_out]
    )

demo.launch(server_name="0.0.0.0", server_port=80, share=True)
