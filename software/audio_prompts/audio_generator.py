#!/usr/bin/env python3

from gtts import gTTS
from pathlib import Path
import subprocess

# Folder to store audio files
AUDIO_DIR = Path(__file__).resolve().parent
AUDIO_DIR.mkdir(exist_ok=True)

def generate_prompt(text: str, name: str, lang="en"):
    """
    Generate Google Translate MP3 and auto-convert to WAV.

    Args:
        text: Text content to speak
        name: Base filename without extension
        lang: Language code (default 'en')
    """

    mp3_path = AUDIO_DIR / f"{name}.mp3"
    wav_path = AUDIO_DIR / f"{name}.wav"

    # --- Step 1: Generate MP3 using gTTS ---
    print(f"[INFO] Generating MP3: {mp3_path}")
    tts = gTTS(text=text, lang=lang)
    tts.save(str(mp3_path))

    # --- Step 2: Convert MP3 to WAV using ffmpeg ---
    print(f"[INFO] Converting to WAV: {wav_path}")
    cmd = [
        "ffmpeg",
        "-y",               # overwrite output files
        "-i", str(mp3_path),
        "-ac", "1",         # mono channel
        "-ar", "16000",     # sample rate for speech models
        str(wav_path)
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    print(f"[DONE] Created:\n  MP3: {mp3_path}\n  WAV: {wav_path}")
    return mp3_path, wav_path


if __name__ == "__main__":
    # Example usage
    generate_prompt(
        text="Welcome! Please try to speak a full sentence or phrase in 3 seconds after the beep sound so we can identify your voice.",
        # text = "Detect new user. Please speak after 'beep' for three times to record your voice.",
        # text = "beep, beep, beep",
        name="voiceprint_welcome_prompt"
    )
