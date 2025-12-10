import sounddevice as sd
import soundfile as sf
import numpy as np
import cv2
import wave
import time
import subprocess
import queue
import json
from pathlib import Path
from vosk import Model, KaldiRecognizer

from database.face_db import FaceDB

# ====== CONFIG ======
MODEL_PATH = "/home/rpi/Documents/project_edmpty/voice_env/vosk-model-small-en-us-0.15"
SAMPLE_RATE = 16000
BLOCK_SIZE = 8000
PROMPT_AUDIO_PATH = Path(__file__).resolve().parents[1] / "audio_prompts" / "welcome_prompt.wav"
END_AUDIO_PATH = Path(__file__).resolve().parents[1] / "audio_prompts" / "recording_end_prompt.wav"
BEEP_AUDIO_PATH = Path(__file__).resolve().parents[1] / "audio_prompts" / "beep.wav"
RESPONSE_WINDOW_SECONDS = 5.0  # Listen for yes/no within 5 seconds after prompt
USER_CHECK_INTERVAL = 0.5  # Check for new users every 0.5 seconds

# Shared SQLite DB (same as camera + Flask)
DB_PATH = Path(__file__).resolve().parents[1] / "database" / "face_database.db"
db = FaceDB(str(DB_PATH))

# Load Vosk model
print("Loading Vosk model...")
model = Model(MODEL_PATH)
recognizer = KaldiRecognizer(model, SAMPLE_RATE)

# Queue for audio data
audio_q = queue.Queue()

# State tracking
last_prompted_user_id = None
prompt_time = None
listening_for_response = False
last_detected_choice = None  # Stores the last yes/no choice (True=yes, False=no, None=no choice yet)

def audio_callback(indata, frames, time_info, status):
    """Callback from sounddevice - push audio into queue."""
    if status:
        print("Audio status:", status)
    audio_q.put(bytes(indata))

def play_prompt_audio(audio_path):
    """Play the welcome prompt WAV file."""
    if not audio_path.exists():
        print(f"Warning: Prompt audio file not found at {audio_path}")
        return
    
    try:
        data, fs = sf.read(str(audio_path))
        
        # Handle stereo to mono conversion if needed
        if len(data.shape) > 1 and data.shape[1] > 1:
            # Convert stereo to mono by averaging channels
            data = np.mean(data, axis=1)
        
        # Convert to float32 if needed (sounddevice prefers float32)
        if data.dtype != np.float32:
            if data.dtype == np.int16:
                # Convert int16 to float32 (-1.0 to 1.0 range)
                data = data.astype(np.float32) / 32767.0
            elif data.dtype == np.float64:
                data = data.astype(np.float32)
            else:
                # For other types, normalize to float32
                data = data.astype(np.float32)
                if np.max(np.abs(data)) > 1.0:
                    data = data / np.max(np.abs(data))
        
        # Play the audio (non-blocking, but we wait for it)
        sd.play(data, samplerate=fs)
        sd.wait()  # Wait until playback is finished
        print("✅ Played prompt")
    except Exception as e:
        print(f"Error playing prompt audio: {e}")


def _set_permission_from_voice(agree: bool):
    """
    Update the 'permission' flag for the last-seen user in the DB
    based on a voice answer. Voice has higher priority than the website
    simply because this write happens later.
    """
    user = db.get_last_seen_user()
    if not user or user.get("user_id") is None:
        print("No active user to update permission for.")
        return

    db.set_permission(user["user_id"], agree)
    print(
        f"Updated permission via voice for user_id={user['user_id']} "
        f"to {'agree (1)' if agree else 'disagree (2)'}."
    )


def on_yes():
    """Store 'yes' choice when detected (will be committed when window expires)."""
    global last_detected_choice
    print("✅ Detected YES!")
    last_detected_choice = True


def on_no():
    """Store 'no' choice when detected (will be committed when window expires)."""
    global last_detected_choice
    print("❌ Detected NO!")
    last_detected_choice = False

def check_and_prompt_user():
    """
    Check if a new user is present and prompt them if needed.
    Returns True if a new user was prompted, False otherwise.
    """
    global last_prompted_user_id, prompt_time, listening_for_response, last_detected_choice
    
    user = db.get_last_seen_user()
    if not user or user.get("user_id") is None:
        # No user present, reset state
        if last_prompted_user_id is not None:
            last_prompted_user_id = None
            listening_for_response = False
            prompt_time = None
            last_detected_choice = None
        return False
    
    current_user_id = user["user_id"]
    
    # If this is a different user, prompt them
    if current_user_id != last_prompted_user_id:
        print(f"New user detected: user_id={current_user_id}, name={user.get('name', 'Unknown')}")
        play_prompt_audio(PROMPT_AUDIO_PATH)
        play_prompt_audio(BEEP_AUDIO_PATH)
        last_prompted_user_id = current_user_id
        prompt_time = time.time()
        listening_for_response = True
        last_detected_choice = None  # Reset choice for new user
        return True
    
    # Check if we're still within the response window
    if listening_for_response and prompt_time is not None:
        elapsed = time.time() - prompt_time
        if elapsed > RESPONSE_WINDOW_SECONDS:
            # Window expired, commit the last detected choice and stop listening
            listening_for_response = False
            if last_detected_choice is not None:
                _set_permission_from_voice(last_detected_choice)
                print(f"Response window expired for user_id={current_user_id}, committed last choice: {'YES' if last_detected_choice else 'NO'}")
            else:
                print(f"Response window expired for user_id={current_user_id}, no response detected")
            last_detected_choice = None  # Reset for next time
            play_prompt_audio(END_AUDIO_PATH)
    
    return False

def main():
    global listening_for_response, prompt_time, last_detected_choice

    print("Starting microphone stream...")
    print(f"Prompt audio path: {PROMPT_AUDIO_PATH}")
    
    last_user_check = 0
    
    with sd.RawInputStream(
        samplerate=SAMPLE_RATE,
        blocksize=BLOCK_SIZE,
        dtype="int16",
        channels=1,
        callback=audio_callback,
    ):
        print("Listening for users and voice responses...")
        while True:
            # Check for new users periodically
            current_time = time.time()
            if current_time - last_user_check >= USER_CHECK_INTERVAL:
                check_and_prompt_user()
                last_user_check = current_time
            
            # Process audio data (non-blocking with timeout)
            try:
                data = audio_q.get(timeout=0.1)
            except queue.Empty:
                continue
            
            if recognizer.AcceptWaveform(data):
                result = recognizer.Result()
                result_dict = json.loads(result)
                text = result_dict.get("text", "").lower()

                if not text:
                    continue

                print("Heard:", text)

                # Only process yes/no if we're listening for a response
                if not listening_for_response:
                    continue
                
                # Check if we're still within the response window
                if prompt_time is not None:
                    elapsed = time.time() - prompt_time
                    if elapsed > RESPONSE_WINDOW_SECONDS:
                        # Window expired, commit the last detected choice
                        listening_for_response = False
                        if last_detected_choice is not None:
                            _set_permission_from_voice(last_detected_choice)
                            print(f"Response window expired during audio processing, committed last choice: {'YES' if last_detected_choice else 'NO'}")
                            last_detected_choice = None
                        play_prompt_audio(END_AUDIO_PATH)
                        continue

                # Simple keyword check - keep listening until window expires
                if any(word in text for word in ["yes", "yeah", "yep", "okay", "sure", "ok"]):
                    on_yes()
                    # Continue listening - don't stop here
                elif any(word in text for word in ["no", "nope"]):
                    on_no()
                    # Continue listening - don't stop here

if __name__ == "__main__":
    main()

