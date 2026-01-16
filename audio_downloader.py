import os
import requests
from urllib.parse import urlparse
from pydub import AudioSegment

URLS_FILE = "urls.txt"
OUTPUT_DIR = "audio_downloads"
OUTPUT_FILE = "story.wav"
TIMEOUT = 30

def download_audios():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    with open(URLS_FILE, "r", encoding="utf-8") as f:
        urls = [line.strip() for line in f if line.strip()]

    for index, url in enumerate(urls, start=1):
        print(f"[{index}/{len(urls)}] Downloading: {url}")

        try:
            response = requests.get(url, stream=True, timeout=TIMEOUT)
            response.raise_for_status()

            ext = ".mp3"
            filename = f"{index:03d}{ext}"
            filepath = os.path.join(OUTPUT_DIR, filename)

            with open(filepath, "wb") as file:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        file.write(chunk)

            print(f"✔ Saved as {filename}")

        except Exception as e:
            print(f"✖ Failed: {e}")

def combine_audios():
    files = sorted(
        f for f in os.listdir(OUTPUT_DIR)
        if f.lower().endswith(".mp3")
    )

    if not files:
        raise RuntimeError("No MP3 files found to combine.")

    combined = AudioSegment.empty()

    for file in files:
        path = os.path.join(OUTPUT_DIR, file)
        print(f"Adding: {file}")
        audio = AudioSegment.from_mp3(path)
        combined += audio

    combined.export(OUTPUT_FILE, format="wav")

    print(f"\n✔ Combined MP3 saved as: {OUTPUT_FILE}")

def main():
    #download_audios()
    combine_audios()

if __name__ == "__main__":
    main()
