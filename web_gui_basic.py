import gradio as gr
import torch
from TTS.api import TTS
import numpy as np
import soundfile as sf
from pydub import AudioSegment, effects
import librosa
import io
import re
import os
import zipfile
import tempfile
from moviepy.editor import ImageClip, CompositeVideoClip, AudioFileClip

USE_GPU = torch.cuda.is_available()

def detect_emotion(text):
    t = text.lower()
    if re.search(r"(scream|blood|dark|shadow|knife|dead|fear|chase)", t):
        return "tense"
    if re.search(r"(cry|alone|sad|tears|empty)", t):
        return "sad"
    return "calm"

def run_pipeline_zip(
    tts_model,
    narrator_speaker,
    dialogue_speaker,
    story_file,
    ambience_file,
    images_zip,
    speed_calm,
    speed_tense,
    speed_sad,
    fade_sec,
    zoom_factor,
    fps,
    device_option,
    output_name
):
    device_gpu = device_option == "GPU"
    tts = TTS(model_name=tts_model, gpu=device_gpu, progress_bar=False)

    with open(story_file.name, "r", encoding="utf-8") as f:
        paragraphs = [p.strip() for p in f.read().split("\n\n") if p.strip()]

    final_audio = AudioSegment.silent(0)

    # Load ambience
    if ambience_file is not None:
        amb = AudioSegment.from_file(ambience_file.name).low_pass_filter(400)
        amb = amb - 20
    else:
        amb = None

    # TTS audio generation
    for para in paragraphs:
        sentences = re.split(r'(?<=[.!?])\s+', para)
        for s in sentences:
            if not s.strip():
                continue
            emotion = detect_emotion(s)
            speed = {
                "calm": speed_calm,
                "tense": speed_tense,
                "sad": speed_sad
            }[emotion]
            speaker = dialogue_speaker if s.startswith(("\"", "“", "'")) else narrator_speaker

            wav = tts.tts(text=s, speaker=speaker)
            wav = np.array(wav, dtype=np.float32)
            wav = librosa.effects.time_stretch(wav, rate=float(speed))
            buf = io.BytesIO()
            sf.write(buf, wav, 24000, format="WAV")
            buf.seek(0)
            seg = AudioSegment.from_file(buf, format="wav")
            final_audio += seg + AudioSegment.silent(250)

    if amb is not None:
        amb_loop = amb * (len(final_audio) // len(amb) + 1)
        final_audio = final_audio.overlay(amb_loop[:len(final_audio)])

    out_audio_path = output_name + ".wav"
    final_audio = effects.normalize(final_audio)
    final_audio.export(out_audio_path, format="wav")

    # ==== Extract images from ZIP ====
    out_video_path = None
    if images_zip is not None:
        with tempfile.TemporaryDirectory() as tmpdir:
            with zipfile.ZipFile(images_zip.name, 'r') as zip_ref:
                zip_ref.extractall(tmpdir)
            images_list = sorted([os.path.join(tmpdir, f)
                                  for f in os.listdir(tmpdir)
                                  if f.lower().endswith((".png",".jpg",".jpeg"))])

            if images_list:
                duration = final_audio.duration_seconds / max(len(images_list), 1)
                clips = []
                t = 0
                for img in images_list:
                    clip = ImageClip(img).set_duration(duration)
                    clip = clip.fadein(fade_sec).fadeout(fade_sec)
                    clip = clip.resize(lambda x: 1 + (zoom_factor - 1)*(x/duration))
                    clip = clip.set_start(t)
                    clips.append(clip)
                    t += duration

                video = CompositeVideoClip(clips)
                audio_clip = AudioFileClip(out_audio_path)
                video = video.set_audio(audio_clip)
                out_video_path = output_name + ".mp4"
                video.write_videofile(out_video_path, fps=fps, codec="libx264", audio_codec="aac")

    return out_audio_path, out_video_path

# ================= GRADIO =================
with gr.Blocks() as demo:
    gr.Markdown("## 🎭 Horror TTS + Video Studio (Web GUI)")

    with gr.Row():
        tts_model = gr.Textbox(label="TTS Model", value="tts_models/en/vctk/vits")
        narrator_speaker = gr.Textbox(label="Narrator Speaker", value="p226")
        dialogue_speaker = gr.Textbox(label="Dialogue Speaker", value="p229")

    story_file = gr.File(label="Story TXT", file_types=[".txt"])
    ambience_file = gr.File(label="Ambience WAV", file_types=[".wav"])
    images_zip = gr.File(label="Images ZIP", file_types=[".zip"])

    with gr.Row():
        speed_calm = gr.Slider(label="Speed Calm", minimum=0.7, maximum=1.4, value=1.0)
        speed_tense = gr.Slider(label="Speed Tense", minimum=0.7, maximum=1.4, value=0.98)
        speed_sad = gr.Slider(label="Speed Sad", minimum=0.7, maximum=1.4, value=0.99)

    fade_sec = gr.Slider(label="Fade Duration (s)", minimum=0, maximum=5, value=1.0)
    zoom_factor = gr.Slider(label="Zoom Factor", minimum=1.0, maximum=2.0, value=1.05)
    fps = gr.Slider(label="Video FPS", minimum=12, maximum=60, value=24, step=1)

    device_option = gr.Dropdown(label="Device", choices=["GPU", "CPU"], value="GPU" if USE_GPU else "CPU")
    output_name = gr.Textbox(label="Output Name", value="story_output")

    run_btn = gr.Button("▶ RUN FULL PIPELINE")
    audio_out = gr.Audio(label="Generated Audio")
    video_out = gr.Video(label="Generated Video")

    run_btn.click(fn=run_pipeline_zip,
                  inputs=[tts_model, narrator_speaker, dialogue_speaker,
                          story_file, ambience_file, images_zip,
                          speed_calm, speed_tense, speed_sad,
                          fade_sec, zoom_factor, fps,
                          device_option, output_name],
                  outputs=[audio_out, video_out])

demo.launch()