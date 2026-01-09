from pydub import AudioSegment
from faster_whisper import WhisperModel
import re
from moviepy.editor import *
from pydub import AudioSegment, effects
from PIL import Image
import subprocess
import random
import math

STORY_FILE = "story.txt"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
AUDIO_MP3 = os.path.join(BASE_DIR, "story.mp3") 
AUDIO_WAV = os.path.join(BASE_DIR, "story.wav") 
IMAGES_PATH = os.path.join(BASE_DIR, "images")   
IMAGES_PROCESSED_PATH = os.path.join(BASE_DIR, "images_processed")    

AI_VIDEOS_PATH = "ai_videos"      
TEMP_AI_VIDEOS_PATH = "temp_clips"

N_SENTENCES = 8                    
LANGUAGE = "en"
TIMED_SENTENCES_FILE = "timed_sentences.txt"

IMG_EXT = ".jpeg"
VIDEO_EXT = ".mp4"

IMAGE_RESOLUTION_WIDTH = 1280
IMAGE_RESOLUTION_HEIGHT = 720
IMAGE_ZOOM = 1
IMAGE_OFFSET = -20

FADE_DURATION = 1                        
ZOOM = 1
OVERSCALE = 1.1
VIDEO_FPS = 60
RENDERING_PRESET = "slow" #ultrafast / slow
VIDEO_RESOLUTION_HEIGHT = "1280"
VIDEO_RESOLUTION_WIDTH = "720"
VIDEO_RESOLUTION = (1280, 720) #"400*240"
VIDEO_ZOOM_FACTOR = 1
VIDEO_BLUR_FACTOR = 1
VIDEO_CODEC = "lanczos"
VIDEO_BITRATE = "30M"
OUTPUT_VIDEO = "final_video.mp4"

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

def convert_to_wav(input_file, output_file):
    audio = AudioSegment.from_mp3(input_file)
    audio.export(output_file, format="wav")

def get_sentences():
    model = WhisperModel(
        "medium",
        device="cuda",
        compute_type="int8_float16"
    )

    segments, info = model.transcribe(
        AUDIO_WAV,
        language=LANGUAGE,
        vad_filter=True,
        word_timestamps=True
    )

    sentences = []
    buffer_words = []
    sentence_start = None

    for seg in segments:
        for w in seg.words:
            if sentence_start is None:
                sentence_start = w.start

            buffer_words.append(w)

            if re.search(r'[.!?]$', w.word):
                sentence_text = " ".join(x.word for x in buffer_words)
                sentence_end = buffer_words[-1].end

                sentences.append({
                    "text": sentence_text.strip(),
                    "start": sentence_start,
                    "end": sentence_end
                })

                buffer_words = []
                sentence_start = None

    if buffer_words:
        sentences.append({
            "text": " ".join(w.word for w in buffer_words).strip(),
            "start": sentence_start,
            "end": buffer_words[-1].end
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
    #files.sort(key=lambda x: os.path.getmtime(os.path.join(folder, x)))
    files.sort(key=lambda x: int(os.path.splitext(x)[0]))
    return [os.path.join(folder, f) for f in files]

def preprocess_images(
    input_dir,
    output_dir,
    target_size=(1280, 720),
    zoom=1.08,
    y_offset=-70,
    ext=(".png", ".jpg", ".jpeg")
):
    os.makedirs(output_dir, exist_ok=True)

    for fname in sorted(os.listdir(input_dir)):
        if not fname.lower().endswith(ext):
            continue

        in_path = os.path.join(input_dir, fname)
        out_path = os.path.join(output_dir, fname)

        img = Image.open(in_path).convert("RGB")
        w, h = img.size
        tw, th = target_size

        zw = int(w * zoom)
        zh = int(h * zoom)

        img = img.resize((zw, zh), Image.LANCZOS)

        x1 = (zw - tw) // 2
        y1 = (zh - th) // 2 + y_offset

        x1 = max(0, min(x1, zw - tw))
        y1 = max(0, min(y1, zh - th))

        img = img.crop((x1, y1, x1 + tw, y1 + th))

        img.save(out_path, quality=95, subsampling=0)

        print(f"Processed: {fname}")

    print("✅ All images processed")

def ease_in_out(t, d):
    return (1 - math.cos(math.pi * t / d)) / 2

def ease_in_out(p):
    return 0.5 - 0.5 * math.cos(math.pi * p)

def linear(p):
    return max(0.0, min(1.0, p))

def smooth_zoom(clip, zoom_start=1.0, zoom_end=1.12, easing='ease_in_out'):
    duration = clip.duration
    def easing_func(t):
            #p = t / duration
            #p = ease_in_out(t, duration)
            p = linear(t / duration)
            if p < 0.5:
                return zoom_start + (zoom_end - zoom_start) * 2 * p * p
            else:
                return zoom_start + (zoom_end - zoom_start) * (1 - 2*(1-p)*(1-p))
        
    return clip.resize(lambda t: easing_func(t))

def zoom_pan_linear(
    img_path,
    duration,
    start_time=0,
    resolution=(1280, 720),
    zoom=1.1,
    direction="right",  # "right", "left", "up", "down"
    fps=60
):
    W, H = resolution

    overscale = zoom * OVERSCALE

    base = (
        ImageClip(img_path)
        .resize(overscale)
        .set_duration(duration)
        .set_fps(fps)
    )

    BW, BH = base.size

    cx = (BW - W) / 2
    cy = (BH - H) / 2

    if direction == "right":
        sx, ex = cx - (BW - W) / 2, cx + (BW - W) / 2
        sy = ey = cy
    elif direction == "left":
        sx, ex = cx + (BW - W) / 2, cx - (BW - W) / 2
        sy = ey = cy
    elif direction == "down":
        sy, ey = cy - (BH - H) / 2, cy + (BH - H) / 2
        sx = ex = cx
    elif direction == "up":
        sy, ey = cy + (BH - H) / 2, cy - (BH - H) / 2
        sx = ex = cx
    else:
        sx = ex = cx
        sy = ey = cy

    def transform(get_frame, t):
        frame = get_frame(t)
        p = min(1, max(0, t / duration))  

        x = sx + (ex - sx) * p
        y = sy + (ey - sy) * p

        return frame[
            int(y):int(y + H),
            int(x):int(x + W)
        ]

    return (
        base.fl(transform, apply_to=["mask"])
        .set_start(start_time)
        .fadein(0.25)
        .fadeout(0.25)
    )


def get_clips(blocks):
    clips = []

    image_files = get_sorted_files(IMAGES_PROCESSED_PATH, IMG_EXT)
    if len(image_files) < len(blocks):
        raise ValueError("Not enough images for the number of blocks.")

    W, H = VIDEO_RESOLUTION

    for idx, block in enumerate(blocks):
        img_path = image_files[idx]  

        duration = block["end"] - block["start"]
        start = block["start"]

        #clip = ImageClip(img_path).resize((int(W * ZOOM), int(H * ZOOM))).set_duration(duration).set_fps(VIDEO_FPS)
        clip = ImageClip(img_path).resize((int(W), int(H))).set_duration(duration).set_fps(VIDEO_FPS)

        #clip = smooth_zoom(clip, zoom_start=1.0, zoom_end=ZOOM)
        #clip = clip.set_position(("center","center"))

        #clip = zoom_pan_linear(img_path=img_path, duration=duration, start_time=start, resolution=(W, H), zoom=ZOOM, direction="right", fps=60)

        clip = clip.fadein(FADE_DURATION).fadeout(FADE_DURATION)
        clip = clip.set_start(start)

        clips.append(clip)

    #video = CompositeVideoClip(clips, size=(W, H))
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
        codec="h264_nvenc",
        audio_codec="aac",
        bitrate=VIDEO_BITRATE,
        threads=0,
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
    #convert_to_wav(AUDIO_MP3, AUDIO_WAV)
    #print("✅ Converted MP3 to WAV.")

    #sentences = get_sentences()
    #print("✅ Extracted sentences from audio.")

    #get_blocks(sentences)
    #print("✅ Created timed sentences file.")

    #preprocess_images(input_dir=IMAGES_PATH, output_dir=IMAGES_PROCESSED_PATH,target_size=(IMAGE_RESOLUTION_WIDTH, IMAGE_RESOLUTION_HEIGHT), zoom=IMAGE_ZOOM, y_offset=IMAGE_OFFSET)
    #print("✅ Preprocessed images for video.")

    blocks = read_blocks()
    #print("✅ Read timed sentences.")

    get_clips(blocks)
    #print("✅ Created video clip from images.")

    #get_clips_from_videos(blocks)
    #print("✅ Created video clip from ai videos.")

if __name__ == "__main__":
    main()