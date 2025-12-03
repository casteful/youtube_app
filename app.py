import torch
from TTS.api import TTS
from TTS.utils.radam import RAdam
from collections import defaultdict
import soundfile as sf
from pydub import AudioSegment
from faster_whisper import WhisperModel
import re
from moviepy.editor import *
from pydub import AudioSegment, effects
import io
import librosa
import numpy as np
import subprocess

torch.serialization.add_safe_globals([RAdam, defaultdict])

STORY_FILE = "story.txt"

TTS_MODEL = "tts_models/en/vctk/vits"
TTS_NARRATOR_SPEAKER = "p262" #p226 p262
TTS_DIALOGUE_SPEAKER = "p262"     
TTS_PRESET = "high_quality"  
TTS_FADE_MS = 400       
TTS_PARAGRAPH_PAUSE_MS = 700       
TTS_BREATH_PAUSE_MS = 250
TTS_AMBIENCE_VOLUME = 20
TTS_AMBIENCE_FILE = "ambience.wav"
TTS_SPEED_CALM = 1   
TTS_SPEED_TENSE = 1
TTS_SPEED_SAD = 1
TTS_TENSE_EMOTION_REGEX = r"(scream|blood|dark|shadow|knife|dead|cold|steps|run|ran|chase|fear|fight|escape|shout|yell|terror|panic|fright|alarmed|nervous|anxious|worried|uneasy|agitated|distressed)"
TTS_SAD_EMOTION_REGEX = r"(tears|cry|empty|alone|silent|lost|lonely|grief|sorrow|heartbreak|regret|mourning|depressed|melancholy|gloomy|despair|hopeless|downcast)"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
AUDIO_MP3 = os.path.join(BASE_DIR, "story.mp3") 
AUDIO_WAV = os.path.join(BASE_DIR, "story.wav") 
IMAGES_PATH = os.path.join(BASE_DIR, "images")    
AI_VIDEOS_PATH = "ai_videos"      
TEMP_AI_VIDEOS_PATH = "temp_clips"

N_SENTENCES = 2                     
LANGUAGE = "en"
TIMED_SENTENCES_FILE = "timed_sentences.txt"

IMG_EXT = ".png"
VIDEO_EXT = ".mp4"

FADE_DURATION = 1                        
ZOOM = 1.2
VIDEO_FPS = 60
RENDERING_PRESET = "slow" #ultrafast
VIDEO_RESOLUTION_HEIGHT = "1280"
VIDEO_RESOLUTION_WIDTH = "720"
VIDEO_ZOOM_FACTOR = 2
VIDEO_BLUR_FACTOR = 1.1
VIDEO_CODEC = "lanczos"
VIDEO_BITRATE = "20M"
OUTPUT_VIDEO = "final_video.mp4"

USE_GPU = torch.cuda.is_available()
tts = TTS(model_name=TTS_MODEL, gpu=USE_GPU, progress_bar=False)

def detect_emotion(text):
    text = text.lower()
    if re.search(TTS_TENSE_EMOTION_REGEX, text):
        return "tense"
    if re.search(TTS_SAD_EMOTION_REGEX, text):
        return "sad"
    return "calm"

def speed_by_emotion(emotion):
    return {
        "tense": TTS_SPEED_TENSE,
        "sad": TTS_SPEED_SAD,
        "calm": TTS_SPEED_CALM
    }[emotion]

def is_dialogue(text):
    return text.strip().startswith(("\"", "“", "'"))

def tts_to_segment(text, speaker, speed):
    wav = tts.tts(
        text=text,
        speaker=speaker
    )

    wav = np.array(wav, dtype=np.float32)
    wav_stretched = librosa.effects.time_stretch(wav, rate=speed)

    buf = io.BytesIO()
    sf.write(buf, wav_stretched, 24000, format="WAV")
    buf.seek(0)
    return AudioSegment.from_file(buf, format="wav")

def retime_video_ffmpeg(src, dst, target_duration):
    result = subprocess.run([
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        src
    ], capture_output=True, text=True)

    original_duration = float(result.stdout.strip())

    pts_factor = target_duration / original_duration

    cmd = [
        "ffmpeg", "-y",
        "-i", src,
        "-filter:v", f"setpts={pts_factor}*PTS",
        "-an",  # no audio here
        "-pix_fmt", "yuv420p",
        dst
    ]

    subprocess.run(cmd, check=True)

def retime_and_upscale_video_ffmpeg(
    input_video,
    output_video,
    target_duration,
    resolution_height=VIDEO_RESOLUTION_HEIGHT,
    resolution_width=VIDEO_RESOLUTION_WIDTH, 
    method=VIDEO_CODEC,
    fps=VIDEO_FPS,
    bitrate=VIDEO_BITRATE
):
    probe_cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "format=duration",
        "-of", "default=nokey=1:noprint_wrappers=1",
        input_video
    ]

    original_duration = float(subprocess.check_output(probe_cmd).decode().strip())
    speed = original_duration / target_duration

    vf = (
        f"hwupload_cuda,"
        f"scale_cuda={resolution_height}:{resolution_width}:interp_algo={method},"
        f"hwdownload,format=nv12,"
        #?f"noise=c0s={0.05}:c0f=t," #noise
        #?f"noise=c0s={0.05}:allf=t," #grain
        #f"minterpolate='fps={fps}:mi_mode=mci:mc_mode=aobmc:me_mode=bidir:vsbmc=1',"
        f"zoompan=z='1+({VIDEO_ZOOM_FACTOR}-1)*(1-cos(on*PI/{fps*target_duration}))/2':d=1:s={resolution_height}x{resolution_width},"
        f"gblur=sigma={VIDEO_BLUR_FACTOR},"
        f"unsharp,"
        f"vignette,"
        f"setpts={1/speed}*PTS"
    )

    cmd = [
        "ffmpeg", "-y",
        "-hwaccel", "cuda",
        "-i", input_video,

        "-vf", vf,
        "-r", str(fps),

        "-c:v", "h264_nvenc",
        "-preset", "p7",           
        "-rc", "vbr",
        "-b:v", bitrate,
        "-maxrate", bitrate,
        "-pix_fmt", "yuv420p",

        "-filter:a", f"atempo={speed}",
        "-c:a", "aac",

        "-movflags", "+faststart",
        output_video
    ]

    subprocess.run(cmd, check=True)

def synthesize_speech(text, output_file):
    with open(STORY_FILE, "r", encoding="utf-8") as f:
        paragraphs = [p.strip() for p in f.read().split("\n\n") if p.strip()]

    ambience = AudioSegment.from_file(TTS_AMBIENCE_FILE).low_pass_filter(400)
    ambience = ambience - TTS_AMBIENCE_VOLUME  

    final_audio = AudioSegment.silent(0)
    for p_i, para in enumerate(paragraphs, 1):
        sentences = re.split(r'(?<=[.!?])\s+', para)
        paragraph_audio = AudioSegment.silent(0)

        for sentence in sentences:
            if not sentence.strip():
                continue
            
            emotion = detect_emotion(sentence)
            speed = speed_by_emotion(emotion)
            
            voice = TTS_DIALOGUE_SPEAKER if is_dialogue(sentence) else TTS_NARRATOR_SPEAKER

            seg = tts_to_segment(sentence, voice, speed)
            paragraph_audio += seg
            paragraph_audio += AudioSegment.silent(TTS_BREATH_PAUSE_MS)
        
        paragraph_audio = paragraph_audio.fade_in(TTS_FADE_MS).fade_out(TTS_FADE_MS)
        final_audio += paragraph_audio
        final_audio += AudioSegment.silent(TTS_PARAGRAPH_PAUSE_MS)

    amb_loop = ambience * (len(final_audio) // len(ambience) + 1)
    amb_loop = amb_loop[:len(final_audio)]
    final_audio = final_audio.overlay(amb_loop)

    final_audio = effects.normalize(final_audio)

    final_audio.export(AUDIO_WAV, format="wav")

def convert_to_wav(input_file, output_file):
    audio = AudioSegment.from_mp3(input_file)
    audio.export(output_file, format="wav")

def get_sentences():
    model = WhisperModel("medium", device="cuda", compute_type="int8_float16")
    segments, info = model.transcribe(
        AUDIO_WAV,
        language=LANGUAGE,
        vad_filter=True
    )

    sentences = []

    for seg in segments:
        parts = re.split(r'(?<=[.!?])\s+', seg.text.strip())
        step = (seg.end - seg.start) / max(len(parts), 1)
        for i, s in enumerate(parts):
            start = seg.start + step * i
            end = seg.start + step * (i + 1)
            sentences.append({
                "text": s,
                "start": start,
                "end": end
            })

    return sentences

def get_blocks(sentences):
    blocks = []
    for i in range(0, len(sentences), N_SENTENCES):
        block = sentences[i:i + N_SENTENCES]
        blocks.append({
            "start": block[0]["start"],
            "end": block[-1]["end"],
            "text": " ".join(s["text"] for s in block)
        })

    with open(TIMED_SENTENCES_FILE, "w", encoding="utf-8") as f:
        for i, b in enumerate(blocks, 1):
            f.write(
                f"BLOCK {i}: {b['start']:.2f} – {b['end']:.2f}\n"
                f"{b['text']}\n\n"
            )

def read_blocks():
    blocks = []
    with open(TIMED_SENTENCES_FILE, "r", encoding="utf-8") as f:
        lines = f.read().splitlines()
    for line in lines:
        if line.startswith("BLOCK"):
            parts = line.split(":")[1].split("–")
            start = float(parts[0].strip())
            end = float(parts[1].strip())
            blocks.append({"start": start, "end": end})
    return blocks

def get_sorted_files(folder, ext):
    files = [f for f in os.listdir(folder) if f.lower().endswith(ext)]
    files.sort(key=lambda x: os.path.getmtime(os.path.join(folder, x)))
    return [os.path.join(folder, f) for f in files]

def smooth_zoom(clip, zoom_start=1.0, zoom_end=1.12, easing='ease_in_out'):
    duration = clip.duration

    def easing_func(t):
        p = t / duration
        if p < 0.5:
            return zoom_start + (zoom_end - zoom_start) * 2 * p * p
        else:
            return zoom_start + (zoom_end - zoom_start) * (1 - 2*(1-p)*(1-p))
    
    return clip.resize(lambda t: easing_func(t))

def smooth_optical_flow(src, dst, target_fps=60):
    cmd = [
        "ffmpeg", "-y",
        "-i", src,
        "-filter_complex", f"minterpolate='mi_mode=mci:mc_mode=aobmc:vsbmc=1:fps={target_fps}'",
        "-pix_fmt", "yuv420p",
        dst
    ]

    subprocess.run(cmd, check=True)

def flash_bloom_flicker(src, dst):
    cmd = [
        "ffmpeg", "-y",
        "-i", src,
        "-filter_complex",
        "split[v1][v2];[v1]gblur=sigma=3[vblur];[v2][vblur]overlay=format=auto:shortest=1,eq=brightness=0.02:saturation=1.1",
        "-pix_fmt", "yuv420p",
        dst
    ]

    subprocess.run(cmd, check=True)

def run_ffmpeg_filter(input_file, output_file, vf_filter):
    cmd = [
        "ffmpeg", "-y", "-i", input_file,
        "-vf", vf_filter,
        "-c:a", "copy",  
        "-pix_fmt", "yuv420p",
        output_file
    ]

    subprocess.run(cmd, check=True)

def get_clips(blocks):
    clips = []

    image_files = get_sorted_files(IMAGES_PATH, IMG_EXT)
    if len(image_files) < len(blocks):
        raise ValueError("Not enough images for the number of blocks.")

    #os.makedirs("temp_clips", exist_ok=True)

    for idx, block in enumerate(blocks):
        img_path = image_files[idx]  

        duration = block["end"] - block["start"]
        start = block["start"]

        #temp_video = f"temp_clips/clip_{idx}.mp4"

        #subprocess.run(
        #    f'ffmpeg -y -loop 1 -i "{img_path}" -t {duration} -vf "fps={VIDEO_FPS},scale={VIDEO_RESOLUTION}" -pix_fmt yuv420p "{temp_video}"',
        #    shell=True,
        #    check=True
        #)

        #clip = VideoFileClip(temp_video)
        #clip = smooth_zoom(clip, zoom_start=1.0, zoom_end=ZOOM)
        clip = ImageClip(img_path).set_duration(duration)
        clip = clip.fadein(FADE_DURATION).fadeout(FADE_DURATION)
        clip = clip.set_fps(VIDEO_FPS)
        clip = clip.resize(lambda t: 1 + (ZOOM-1)*(t/duration))
        clip = clip.set_position(("center","center"))
        clip = clip.set_start(start)
        clips.append(clip)

    video = CompositeVideoClip(clips)
    audio = AudioFileClip(AUDIO_WAV)
    video = video.set_audio(audio)

    video.write_videofile(
        OUTPUT_VIDEO,
        fps=VIDEO_FPS,
        codec="libx264",
        audio_codec="aac",
        bitrate=VIDEO_BITRATE,
        preset=RENDERING_PRESET,
        threads=8,
        ffmpeg_params=["-pix_fmt", "yuv420p", "-vsync", "0", "-movflags", "+faststart"]
    )

    audio.close()
    video.close()

def get_clips_from_videos(blocks):
    clips = []

    video_files = get_sorted_files(AI_VIDEOS_PATH, VIDEO_EXT)

    if len(video_files) < len(blocks):
        raise ValueError("Not enough AI videos for blocks.")

    os.makedirs(TEMP_AI_VIDEOS_PATH, exist_ok=True)

    for i, block in enumerate(blocks):
        start = block["start"]
        end = block["end"]
        duration = end - start  

        src_video = video_files[i]
        temp_retime = f"{TEMP_AI_VIDEOS_PATH}/retimed_{i:03d}.mp4"
        retime_and_upscale_video_ffmpeg(src_video, temp_retime, duration)

        clip = VideoFileClip(temp_retime).set_start(start)
        clip = clip.fadein(FADE_DURATION).fadeout(FADE_DURATION)
        clips.append(clip)

    video = CompositeVideoClip(clips)
    audio = AudioFileClip(AUDIO_WAV)
    video = video.set_audio(audio)

    video.write_videofile(
        OUTPUT_VIDEO,
        fps=VIDEO_FPS,
        codec="h264_nvenc",   # ✅ GPU encoder. CPU - "libx264"
        audio_codec="aac",
        bitrate=VIDEO_BITRATE,
        threads=0,            # let NVENC manage
        ffmpeg_params=[
            "-pix_fmt", "yuv420p",
            "-rc", "vbr",
            "-movflags", "+faststart",
            "-vsync", "0"
        ]
    )

    audio.close()
    video.close()
    
def main():
    #synthesize_speech(STORY_FILE, AUDIO_WAV)
    print("✅ Synthesized speech from text.")

    convert_to_wav(AUDIO_MP3, AUDIO_WAV)
    print("✅ Converted MP3 to WAV.")

    sentences = get_sentences()
    print("✅ Extracted sentences from audio.")

    get_blocks(sentences)
    print("✅ Created timed sentences file.")

    blocks = read_blocks()
    print("✅ Read timed sentences.")

    #get_clips(blocks)
    #print("✅ Created video clip from images.")

    get_clips_from_videos(blocks)
    print("✅ Created video clip from ai videos.")

if __name__ == "__main__":
    main()