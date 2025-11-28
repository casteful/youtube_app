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

torch.serialization.add_safe_globals([RAdam, defaultdict])

STORY_FILE = "story.txt"

TTS_MODEL = "tts_models/en/vctk/vits"
TTS_NARRATOR_SPEAKER = "p229" #p226 p262
TTS_DIALOGUE_SPEAKER = "p229"     
TTS_PRESET = "high_quality"  
TTS_FADE_MS = 400       
TTS_PARAGRAPH_PAUSE_MS = 700       
TTS_BREATH_PAUSE_MS = 250
TTS_AMBIENCE_VOLUME = 20
TTS_AMBIENCE_FILE = "ambience.wav"
TTS_SPEED_CALM = 1   
TTS_SPEED_TENSE = 0.98
TTS_SPEED_SAD = 0.99
TTS_TENSE_EMOTION_REGEX = r"(scream|blood|dark|shadow|knife|dead|cold|steps|run|ran|chase|fear|fight|escape|shout|yell|terror|panic|fright|alarmed|nervous|anxious|worried|uneasy|agitated|distressed)"
TTS_SAD_EMOTION_REGEX = r"(tears|cry|empty|alone|silent|lost|lonely|grief|sorrow|heartbreak|regret|mourning|depressed|melancholy|gloomy|despair|hopeless|downcast)"

AUDIO_MP3 = r"C:\Users\Fred\Desktop\youtube_app\story.mp3" 
AUDIO_WAV = r"C:\Users\Fred\Desktop\youtube_app\story.wav" 
IMAGES_PATH = r"C:\Users\Fred\Desktop\youtube_app\images"    

N_SENTENCES = 3                     
LANGUAGE = "en"
TIMED_SENTENCES_FILE = "timed_sentences.txt"

FADE_DURATION = 1                        
ZOOM = 1.05
VIDEO_FPS = 24
OUTPUT_VIDEO = "final_video.mp4"

USE_GPU = torch.cuda.is_available()
print(USE_GPU)
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

def synthesize_speech(text, output_file):
    with open(STORY_FILE, "r", encoding="utf-8") as f:
        paragraphs = [p.strip() for p in f.read().split("\n\n") if p.strip()]

    ambience = AudioSegment.from_file(TTS_AMBIENCE_FILE).low_pass_filter(400)
    ambience = ambience - TTS_AMBIENCE_VOLUME  

    final_audio = AudioSegment.silent(0)
    for p_i, para in enumerate(paragraphs, 1):
        print(f"Processing paragraph {p_i}/{len(paragraphs)}")

        sentences = re.split(r'(?<=[.!?])\s+', para)
        paragraph_audio = AudioSegment.silent(0)

        for sentence in sentences:
            if not sentence.strip():
                continue
            
            emotion = detect_emotion(sentence)
            speed = speed_by_emotion(emotion)
            
            voice = TTS_DIALOGUE_SPEAKER if is_dialogue(sentence) else TTS_NARRATOR_SPEAKER
            print(f"  Sentence: \"{sentence}\" | Emotion: {emotion} | Speed: {speed:.2f} | Voice: {voice}")
            
            seg = tts_to_segment(sentence, voice, speed)
            paragraph_audio += seg
            paragraph_audio += AudioSegment.silent(TTS_BREATH_PAUSE_MS)
        
        paragraph_audio = paragraph_audio.fade_in(TTS_FADE_MS).fade_out(TTS_FADE_MS)
        final_audio += paragraph_audio
        final_audio += AudioSegment.silent(TTS_PARAGRAPH_PAUSE_MS)

    print("Mixing ambience...")
    amb_loop = ambience * (len(final_audio) // len(ambience) + 1)
    amb_loop = amb_loop[:len(final_audio)]
    final_audio = final_audio.overlay(amb_loop)

    print("Normalizing loudness...")
    final_audio = effects.normalize(final_audio)

    final_audio.export(AUDIO_WAV, format="wav")
    print("\n✅ MASTER COMPLETE:", AUDIO_WAV)

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
        print(seg)
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

def get_clips(blocks):
    clips = []

    for idx, block in enumerate(blocks):
        img_path = f"{IMAGES_PATH}\\{idx+1}.png"  

        duration = block["end"] - block["start"]
        start = block["start"]

        clip = ImageClip(img_path).set_duration(duration)
        clip = clip.fadein(FADE_DURATION).fadeout(FADE_DURATION)
        zoom_factor = ZOOM
        clip = clip.resize(lambda t: 1 + (zoom_factor-1)*(t/duration))
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
        preset="ultrafast",
        threads=8
    )

    audio.close()
    
def main():
    synthesize_speech(STORY_FILE, AUDIO_WAV)
    print("✅ Synthesized speech from text.")
    #convert_to_wav(AUDIO_MP3, AUDIO_WAV)
    print("✅ Converted MP3 to WAV.")
    #sentences = get_sentences()
    print("✅ Extracted sentences from audio.")
    #get_blocks(sentences)
    print("✅ Created timed sentences file.")
    #blocks = read_blocks()
    print("✅ Read timed sentences.")
    #get_clips(blocks)
    print("✅ Created video clip from images.")

if __name__ == "__main__":
    main()