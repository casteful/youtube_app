import gradio as gr
import torch
from TTS.api import TTS
import numpy as np
import soundfile as sf
from pydub import AudioSegment, effects
import io
import os
import re
from moviepy.editor import ImageClip, CompositeVideoClip, AudioFileClip

# ================= CONFIG =================
USE_GPU = torch.cuda.is_available()

# ================= EMOTION DETECTION =================
def detect_emotion(text):
    text = text.lower()
    if re.search(r"(scream|blood|dark|shadow|knife|dead|fear|chase)", text):
        return "tense"
    if re.search(r"(cry|alone|sad|tears|empty)", text):
        return "sad"
    return "calm"

# ================= TTS + AUDIO =================
def synthesize_tts(
    story_text,
    tts_model_name,
    narrator_speaker,
    dialogue_speaker,
    speed_calm,
    speed_tense,
    speed_sad,
    ambience_file,
    fade_sec,
    zoom,
    fps,
    images_zip
):
    # Load TTS model
    tts = TTS(model_name=tts_model_name, gpu=USE_GPU, progress_bar=False)

    # Split paragraphs
    paragraphs = [p.strip() for p in story_text.split("\n\n") if p.strip()]
    final_audio = AudioSegment.silent(0)

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

            # Speed adjustment
            if speed != 1.0:
                import librosa
                wav = librosa.effects.time_stretch(wav, rate=speed)

            buf = io.BytesIO()
            sf.write(buf, wav, 24000, format="WAV")
            buf.seek(0)
            seg = AudioSegment.from_file(buf, format="wav")
            final_audio += seg + AudioSegment.silent(250)

    # Mix ambience if provided
    if ambience_file:
        amb = AudioSegment.from_file(ambience_file.name).low_pass_filter(400)
        amb = amb - 20
        amb_loop = amb * (len(final_audio) // len(amb) + 1)
        final_audio = final_audio.overlay(amb_loop[:len(final_audio)])

    # Normalize
    final_audio = effects.normalize(final_audio)

    # Save audio
    audio_path = "story_output.wav"
    final_audio.export(audio_path, format="wav")

    # ===== Video generation if images uploaded =====
    video_path = None
    if images_zip:
        import zipfile
        import tempfile

        tmp_dir = tempfile.mkdtemp()
        with zipfile.ZipFile(images_zip.name, 'r') as zip_ref:
            zip_ref.extractall(tmp_dir)

        images_list = sorted([
            os.path.join(tmp_dir, f)
            for f in os.listdir(tmp_dir)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ])

        if images_list:
            clips = []
            duration = final_audio.duration_seconds / max(len(images_list),1)
            t = 0
            for img in images_list:
                clip = ImageClip(img).set_duration(duration)
                clip = clip.fadein(fade_sec).fadeout(fade_sec)
                clip = clip.resize(lambda x: 1 + (zoom-1)*(x/duration))
                clip = clip.set_start(t)
                clips.append(clip)
                t += duration

            video = CompositeVideoClip(clips)
            audio_clip = AudioFileClip(audio_path)
            video = video.set_audio(audio_clip)
            video_path = "story_output.mp4"
            video.write_videofile(video_path, fps=fps, codec="libx264", audio_codec="aac")

    return (audio_path, video_path) if video_path else audio_path

# ================= GRADIO UI =================
with gr.Blocks() as demo:
    gr.Markdown("## Horror TTS + Video Studio (Local GPU + ngrok)")

    with gr.Row():
        story_text = gr.Textbox(label="Story Text", lines=8, placeholder="Paste your story here...")

    with gr.Row():
        tts_model_name = gr.Dropdown(label="TTS Model", choices=[
            "tts_models/en/vctk/vits",
            "tts_models/multilingual/multi-dataset/xtts_v2",
            "tts_models/en/ljspeech/speedy-speech"
        ], value="tts_models/en/vctk/vits")
        narrator_speaker = gr.Textbox(label="Narrator Speaker", value="p226")
        dialogue_speaker = gr.Textbox(label="Dialogue Speaker", value="p229")

    with gr.Row():
        speed_calm = gr.Slider(label="Speed Calm", minimum=0.7, maximum=1.4, value=1.0)
        speed_tense = gr.Slider(label="Speed Tense", minimum=0.7, maximum=1.4, value=0.98)
        speed_sad = gr.Slider(label="Speed Sad", minimum=0.7, maximum=1.4, value=0.99)

    with gr.Row():
        ambience_file = gr.File(label="Ambience WAV", file_types=[".wav"])
        fade_sec = gr.Slider(label="Fade (s)", minimum=0.1, maximum=5.0, value=1.0)
        zoom = gr.Slider(label="Zoom", minimum=1.0, maximum=2.0, value=1.05)
        fps = gr.Slider(label="FPS", minimum=15, maximum=60, value=24, step=1)

    with gr.Row():
        images_zip = gr.File(label="Images Folder (Zip)", file_types=[".zip"])

    run_btn = gr.Button("▶ RUN FULL PIPELINE")
    output_audio = gr.Audio(label="Generated Audio")
    output_video = gr.Video(label="Generated Video")

    run_btn.click(
        fn=synthesize_tts,
        inputs=[
            story_text, tts_model_name, narrator_speaker, dialogue_speaker,
            speed_calm, speed_tense, speed_sad, ambience_file,
            fade_sec, zoom, fps, images_zip
        ],
        outputs=[output_audio, output_video]
    )

demo.launch(server_name="0.0.0.0", server_port=80)