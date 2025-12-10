
"""
Unified voice recognition and enrollment pipeline.
Records voice sample, identifies or enrolls user, and requests permission for new users.
"""

import sounddevice as sd
import soundfile as sf
import numpy as np
import queue
import json
import time
from pathlib import Path
from vosk import Model, KaldiRecognizer
from database.face_db import FaceDB

# Try to import Resemblyzer first (best for speaker recognition)
try:
    from resemblyzer import VoiceEncoder, preprocess_wav
    RESEMBLYZER_AVAILABLE = True
    print("✓ Resemblyzer available - using speaker embeddings for accurate recognition")
except ImportError:
    RESEMBLYZER_AVAILABLE = False
    print("⚠️  Resemblyzer not available. Install with: pip install resemblyzer")
    print("   Falling back to MFCC-based recognition (less accurate)")

# Fallback to MFCC if Resemblyzer not available
if not RESEMBLYZER_AVAILABLE:
    try:
        import librosa
        USE_LIBROSA = True
    except ImportError:
        try:
            from python_speech_features import mfcc
            USE_LIBROSA = False
        except ImportError:
            print("ERROR: Need either 'resemblyzer' or 'librosa' or 'python_speech_features' installed.")
            print("Install with: pip install resemblyzer  (recommended)")
            print("   OR: pip install librosa  OR  pip install python_speech_features")
            import sys
            sys.exit(1)

# ====== CONFIG ======
# Try multiple possible paths for Vosk model
VOSK_MODEL_PATHS = [
    Path(__file__).parent / "vosk-model-small-en-us-0.15",
    Path("/home/rpi/Documents/project/voice_env/vosk-model-small-en-us-0.15"),
    Path("/home/rpi/Documents/project_edmpty/database/../voice_env/vosk-model-small-en-us-0.15"),
]

SAMPLE_RATE = 16000
BLOCK_SIZE = 8000
RECORD_DURATION = 3.0  # Seconds - Resemblyzer works well with 3+ seconds, MFCC needs longer
MFCC_COEFFS = 13
N_MELS = 40
# Threshold depends on method: cosine similarity for Resemblyzer, L2 for MFCC
if RESEMBLYZER_AVAILABLE:
    IDENTIFICATION_THRESHOLD = 0.75  # Cosine similarity (0.0-1.0, higher = more strict)
    USE_COSINE = True
else:
    IDENTIFICATION_THRESHOLD = 0.4  # L2 distance (lower = more strict)
    USE_COSINE = False
MIN_CONFIDENCE_DIFF = 0.15  # Minimum difference between best and second-best match for confidence
VERIFICATION_SAMPLES = 2  # Number of samples to verify identity (reduces false positives)
ENROLLMENT_SAMPLES = 3 if RESEMBLYZER_AVAILABLE else 5  # Resemblyzer needs fewer samples

# Audio prompt paths
PROMPT_AUDIO_PATH = Path(__file__).resolve().parents[1] / "audio_prompts" / "welcome_prompt.wav"
NEW_USER_AUDIO_PATH = Path(__file__).resolve().parents[1] / "audio_prompts" / "voiceprint_new_user_prompt.wav"
BEEP_AUDIO_PATH = Path(__file__).resolve().parents[1] / "audio_prompts" / "beep.wav"
WELCOME_AUDIO_PATH = Path(__file__).resolve().parents[1] / "audio_prompts" / "voiceprint_welcome_prompt.wav"

RESPONSE_WINDOW_SECONDS = 5.0  # Listen for yes/no within 5 seconds after prompt

DB_PATH = Path(__file__).resolve().parents[1] / "database" / "face_database.db"
db = FaceDB(str(DB_PATH))

# Initialize Resemblyzer encoder if available
encoder = None
if RESEMBLYZER_AVAILABLE:
    print("Loading Resemblyzer voice encoder...")
    encoder = VoiceEncoder()
    print("✓ Resemblyzer encoder loaded")

# Load Vosk model for speech recognition
print("Loading Vosk model...")
model = None
recognizer = None
VOSK_AVAILABLE = False

for model_path in VOSK_MODEL_PATHS:
    if model_path.exists():
        try:
            model = Model(str(model_path))
            recognizer = KaldiRecognizer(model, SAMPLE_RATE)
            VOSK_AVAILABLE = True
            print(f"✓ Loaded Vosk model from: {model_path}")
            break
        except Exception as e:
            print(f"Warning: Could not load Vosk model from {model_path}: {e}")
            continue

if not VOSK_AVAILABLE:
    print("⚠️  Warning: Could not load Vosk model from any path.")
    print("   Will skip speech recognition for permission questions.")
    print("   Available paths tried:")
    for p in VOSK_MODEL_PATHS:
        print(f"     - {p} (exists: {p.exists()})")

# Queue for audio data
audio_q = queue.Queue()


def extract_speaker_embedding_resemblyzer(audio: np.ndarray) -> np.ndarray:
    """
    Extract speaker embedding using Resemblyzer (best for speaker recognition).
    Returns a 256-dimensional vector representing the speaker's voice.
    Works well with single sentences!
    """
    if not RESEMBLYZER_AVAILABLE or encoder is None:
        return None
    
    # Resemblyzer expects float32 audio in range [-1, 1]
    if audio.dtype == np.int16:
        audio_float = audio.astype(np.float32) / 32768.0
    else:
        audio_float = audio.astype(np.float32)
    
    # Preprocess audio for Resemblyzer
    wav = preprocess_wav(audio_float, source_sr=SAMPLE_RATE)
    
    # Extract speaker embedding (256 dimensions)
    embedding = encoder.embed_utterance(wav)
    
    return embedding.astype(np.float32)


def extract_speaker_features(audio: np.ndarray, sr: int = SAMPLE_RATE) -> np.ndarray:
    """
    Extract speaker-specific features that are less dependent on speech content.
    Combines MFCC with pitch, spectral centroid, and other speaker characteristics.
    """
    if USE_LIBROSA:
        if audio.dtype != np.float32:
            if audio.dtype == np.int16:
                audio = audio.astype(np.float32) / 32768.0
            else:
                audio = audio.astype(np.float32)
        
        # MFCC features
        mfccs = librosa.feature.mfcc(
            y=audio,
            sr=sr,
            n_mfcc=MFCC_COEFFS,
            n_mels=N_MELS,
            hop_length=512,
            n_fft=2048
        )
        mfcc_mean = np.mean(mfccs, axis=1)
        
        # Speaker-specific features (less dependent on words)
        # Pitch (fundamental frequency) - speaker characteristic
        try:
            pitches, magnitudes = librosa.piptrack(y=audio, sr=sr, threshold=0.1)
            # Extract non-zero pitch values from the 2D array
            pitch_values = []
            for t in range(pitches.shape[1]):
                index = magnitudes[:, t].argmax()
                pitch = pitches[index, t]
                if pitch > 0:
                    pitch_values.append(pitch)
            pitch_values = np.array(pitch_values)
            pitch_mean = np.mean(pitch_values) if len(pitch_values) > 0 else 0.0
            pitch_std = np.std(pitch_values) if len(pitch_values) > 0 else 0.0
        except Exception:
            # Fallback if pitch detection fails
            pitch_mean = 0.0
            pitch_std = 0.0
        
        # Spectral centroid - brightness of sound (speaker characteristic)
        spectral_centroids = librosa.feature.spectral_centroid(y=audio, sr=sr)[0]
        centroid_mean = np.mean(spectral_centroids)
        centroid_std = np.std(spectral_centroids)
        
        # Zero crossing rate - voice quality indicator
        zcr = librosa.feature.zero_crossing_rate(audio)[0]
        zcr_mean = np.mean(zcr)
        
        # Spectral rolloff - frequency below which 85% of energy is contained
        rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr)[0]
        rolloff_mean = np.mean(rolloff)
        
        # Combine all features
        speaker_features = np.concatenate([
            mfcc_mean,
            np.array([pitch_mean, pitch_std, centroid_mean, centroid_std, zcr_mean, rolloff_mean])
        ])
        
        return speaker_features.astype(np.float32)
    else:
        # Fallback to basic MFCC if librosa not available
        return extract_mfcc_speech_features(audio, sr)


def extract_mfcc_librosa(audio: np.ndarray, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Extract MFCC features using librosa (kept for backward compatibility)."""
    return extract_speaker_features(audio, sr)


def extract_mfcc_speech_features(audio: np.ndarray, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Extract MFCC features using python_speech_features."""
    if audio.dtype == np.int16:
        audio = audio.astype(np.float32) / 32768.0
    
    mfccs = mfcc(
        audio,
        samplerate=sr,
        numcep=MFCC_COEFFS,
        nfilt=N_MELS,
        nfft=2048
    )
    return np.mean(mfccs, axis=0).astype(np.float32)


def extract_mfcc(audio: np.ndarray, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Extract MFCC features using available library."""
    if USE_LIBROSA:
        # Use speaker-specific features if librosa available
        return extract_speaker_features(audio, sr)
    else:
        return extract_mfcc_speech_features(audio, sr)

def play_prompt_audio(audio_path):
    """Play the prompt WAV file."""
    if not audio_path.exists():
        print(f"Warning: Prompt audio file not found at {audio_path}")
        return False
    
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
        print("✅ Played audio prompt")
        return True
    except Exception as e:
        print(f"Error playing prompt audio: {e}")
        return False


def record_audio(duration: float = RECORD_DURATION) -> np.ndarray:
    """Record audio for specified duration."""
    audio = sd.rec(
        int(duration * SAMPLE_RATE),
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype='int16'
    )
    sd.wait()
    return audio.flatten()


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Calculate cosine similarity between two vectors."""
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot_product / (norm_a * norm_b)


def find_top_matches(query_encoding: np.ndarray, embed_dim: int, top_n: int = 3, use_cosine: bool = False) -> list:
    """
    Find top N matches with their scores for confidence checking.
    Only compares embeddings of the same dimension to avoid dimension mismatch errors.
    
    Parameters
    ----------
    use_cosine : bool
        If True, use cosine similarity (for Resemblyzer). If False, use L2 distance (for MFCC).
    
    Returns
    -------
    list of dicts with keys: user_id, voice_id, score, sorted by score (best first)
    """
    matches = []
    
    # Get all candidates and calculate distances
    # Use embed_dim=0 to infer dimension from blob, then filter by matching dimensions
    cur = db.con.cursor()
    cur.execute(
        f"SELECT voice_id, user_id, {db.VOICE_VEC_COL} FROM {db.VOICE_TABLE}"
    )
    
    for voice_id, user_id, blob in cur.fetchall():
        # Infer dimension from blob size (float32 = 4 bytes)
        stored_dim = len(blob) // 4
        
        # Only compare if dimensions match
        if stored_dim != embed_dim:
            continue  # Skip embeddings with different dimensions
        
        # Decode the embedding
        try:
            enc = db._blob_to_enc(blob, stored_dim)
        except ValueError as e:
            print(f"⚠️  Warning: Could not decode embedding for voice_id={voice_id}: {e}")
            continue
        
        # Calculate similarity/distance
        if use_cosine:
            # Cosine similarity (higher is better)
            score = cosine_similarity(query_encoding, enc)
        else:
            # L2 distance (lower is better)
            score = db._l2(query_encoding, enc)
        
        matches.append({
            "user_id": user_id,
            "voice_id": voice_id,
            "score": score
        })
    
    # Sort by score
    if use_cosine:
        matches.sort(key=lambda x: x["score"], reverse=True)  # Higher is better
        # Return only matches above threshold
        valid_matches = [m for m in matches if m["score"] >= IDENTIFICATION_THRESHOLD]
    else:
        matches.sort(key=lambda x: x["score"])  # Lower is better
        # Return only matches below threshold
        valid_matches = [m for m in matches if m["score"] <= IDENTIFICATION_THRESHOLD]
    
    return valid_matches[:top_n]


def identify_speaker(audio: np.ndarray, require_verification: bool = True) -> dict:
    """
    Identify the speaker from audio. Works with a single sentence!
    
    Uses Resemblyzer if available (best for speaker recognition), 
    otherwise falls back to MFCC-based method.
    
    Parameters
    ----------
    audio : np.ndarray
        Audio sample to identify (can be a single sentence)
    require_verification : bool
        If True, requires multiple samples to verify (reduces false positives)
    
    Returns
    -------
    dict with keys: user_id, name, voice_id, score, confidence, or None if no match
    """
    # Try Resemblyzer first (best for speaker recognition)
    if RESEMBLYZER_AVAILABLE and encoder is not None:
        feature_vec = extract_speaker_embedding_resemblyzer(audio)
        if feature_vec is not None:
            embed_dim = int(feature_vec.size)
            top_matches = find_top_matches(feature_vec, embed_dim, top_n=3, use_cosine=True)
            
            if len(top_matches) == 0:
                # Check if there are any embeddings in the database at all
                cur = db.con.cursor()
                cur.execute(f"SELECT COUNT(*) FROM {db.VOICE_TABLE}")
                count = cur.fetchone()[0]
                if count > 0:
                    # There are embeddings, but none match - likely dimension mismatch
                    print("⚠️  No matching embeddings found. Users may need to re-enroll with Resemblyzer.")
                    print(f"   (Looking for {embed_dim}-dim embeddings, but database may have different dimensions)")
                return None
            
            best_match = top_matches[0]
            
            # Confidence check for cosine similarity
            confidence_score = best_match["score"]
            confidence_level = "high"
            
            if len(top_matches) > 1:
                score_diff = best_match["score"] - top_matches[1]["score"]
                if score_diff < 0.1:
                    confidence_level = "medium"
                if score_diff < 0.05 or confidence_score < 0.80:
                    confidence_level = "low"
                    if confidence_score < 0.75:  # Below threshold
                        return None
            
            # Get user info
            cur = db.con.cursor()
            cur.execute(
                f"SELECT name, permission FROM {db.USER_TABLE} WHERE user_id = ?",
                (best_match["user_id"],)
            )
            r = cur.fetchone()
            
            result = {
                "user_id": best_match["user_id"],
                "voice_id": best_match["voice_id"],
                "score": best_match["score"],
                "name": r[0] if r else None,
                "permission": bool(int(r[1])) if r and r[1] is not None else False,
                "confidence": confidence_level
            }
            
            return result
    
    # Fallback to MFCC-based method
    if USE_LIBROSA:
        feature_vec = extract_speaker_features(audio)
    else:
        feature_vec = extract_mfcc(audio)
    
    embed_dim = int(feature_vec.size)
    
    # Find top matches for confidence checking
    top_matches = find_top_matches(feature_vec, embed_dim, top_n=3, use_cosine=False)
    
    if len(top_matches) == 0:
        return None
    
    best_match = top_matches[0]
    
    # Confidence check: ensure best match is significantly better than second-best
    if len(top_matches) > 1:
        score_diff = top_matches[1]["score"] - best_match["score"]
        if score_diff < MIN_CONFIDENCE_DIFF:
            # Scores are too close - not confident enough
            print(f"⚠️  Low confidence: best score={best_match['score']:.4f}, "
                  f"second={top_matches[1]['score']:.4f}, diff={score_diff:.4f}")
            return None
    
    # Get user info
    cur = db.con.cursor()
    cur.execute(
        f"SELECT name, permission FROM {db.USER_TABLE} WHERE user_id = ?",
        (best_match["user_id"],)
    )
    r = cur.fetchone()
    
    result = {
        "user_id": best_match["user_id"],
        "voice_id": best_match["voice_id"],
        "score": best_match["score"],
        "name": r[0] if r else None,
        "permission": bool(int(r[1])) if r and r[1] is not None else False,
        "confidence": "high" if len(top_matches) == 1 or (len(top_matches) > 1 and 
                  (top_matches[1]["score"] - best_match["score"]) >= MIN_CONFIDENCE_DIFF) else "medium"
    }
    
    return result


def enroll_new_user(name: str, num_samples: int = ENROLLMENT_SAMPLES) -> int:
    """
    Enroll a new user by recording multiple voice samples and storing the average MFCC template.
    
    Parameters
    ----------
    name : str
        Name of the user to enroll
    num_samples : int
        Number of voice samples to record
        
    Returns
    -------
    int: user_id of the enrolled user
    """
    print(f"\n{'='*50}")
    print(f"Enrolling new user: {name}")
    print(f"Recording {num_samples} samples of {RECORD_DURATION} seconds each")
    print(f"{'='*50}\n")
    
    play_audio = play_prompt_audio(NEW_USER_AUDIO_PATH)
    
    mfcc_samples = []
    
    for i in range(num_samples):
        print(f"\nSample {i+1}/{num_samples}")
        print("⚠️  IMPORTANT: Please say a FULL SENTENCE or multiple words.")
        print("   Do NOT just say 'yes' or 'no' - say something like:")
        print("   'Hello, my name is [your name]' or 'I am enrolling my voice'")
        print("   This helps distinguish your voice from others.")
        print("\nGet ready...")
        time.sleep(1)
        print("Speak now!")
        
        # play the beep sound to remind the user
        play_audio = play_prompt_audio(BEEP_AUDIO_PATH)
        
        audio = record_audio(RECORD_DURATION)
        
        # Extract speaker embedding/features
        if RESEMBLYZER_AVAILABLE and encoder is not None:
            feature_vec = extract_speaker_embedding_resemblyzer(audio)
        else:
            feature_vec = extract_mfcc(audio)
        mfcc_samples.append(feature_vec)
        
        print(f"✓ Recorded sample {i+1}, Feature shape: {feature_vec.shape}")
    
    # Average all samples to create template
    template = np.mean(mfcc_samples, axis=0).astype(np.float32)
    print(f"\n✓ Template vector shape: {template.shape}")
    print(f"✓ Template vector range: [{template.min():.2f}, {template.max():.2f}]")
    
    # Store in database
    voice_id = db.enroll_voice(name=name, encoding=template)
    user = db.get_user_by_name(name)
    user_id = user["user_id"] if user else None
    
    print(f"\n✅ Successfully enrolled '{name}' with voice_id={voice_id}, user_id={user_id}")
    print(f"   Template stored in database.\n")
    
    return user_id


def audio_callback(indata, frames, time_info, status):
    """Callback from sounddevice - push audio into queue."""
    if status:
        print("Audio status:", status)
    audio_q.put(bytes(indata))


def detect_wake_word(wake_phrase: str = "hi cursor", timeout: float = None) -> bool:
    """
    Continuously listen for the wake word/phrase.
    
    Parameters
    ----------
    wake_phrase : str
        The wake phrase to detect (default: "hi cursor")
    timeout : float, optional
        Maximum time to listen in seconds. If None, listens indefinitely.
    
    Returns
    -------
    bool
        True if wake word detected, False if timeout reached
    """
    if not VOSK_AVAILABLE or recognizer is None:
        print("⚠️  Vosk not available. Cannot detect wake word.")
        return False
    
    print(f"\n{'='*50}")
    print(f"🔇 System is sleeping...")
    print(f"   Listening for wake word: '{wake_phrase}'")
    print(f"   Say '{wake_phrase}' to activate the system")
    print(f"{'='*50}\n")
    
    start_time = time.time()
    
    # Clear the audio queue
    while not audio_q.empty():
        try:
            audio_q.get_nowait()
        except queue.Empty:
            break
    
    # Reset recognizer
    recognizer.Reset()
    
    # Buffer to store recent words for phrase matching
    recent_words = []
    max_words_buffer = 10  # Keep last 10 words
    
    while True:
        # Check timeout
        if timeout is not None and (time.time() - start_time) > timeout:
            return False
        
        try:
            # Get audio data from queue
            data = audio_q.get(timeout=0.1)
            
            # Process with Vosk
            if recognizer.AcceptWaveform(data):
                result = recognizer.Result()
                result_dict = json.loads(result)
                text = result_dict.get("text", "").strip().lower()
                
                if text:
                    words = text.split()
                    recent_words.extend(words)
                    # Keep only last N words
                    if len(recent_words) > max_words_buffer:
                        recent_words = recent_words[-max_words_buffer:]
                    
                    # Check if wake phrase is in recent words
                    recent_text = " ".join(recent_words)
                    if wake_phrase.lower() in recent_text:
                        print(f"\n✅ Wake word detected: '{wake_phrase}'")
                        return True
            else:
                # Partial result (still speaking)
                partial = recognizer.PartialResult()
                partial_dict = json.loads(partial)
                partial_text = partial_dict.get("partial", "").strip().lower()
                
                if partial_text:
                    # Check partial text for wake phrase
                    if wake_phrase.lower() in partial_text:
                        print(f"\n✅ Wake word detected (partial): '{wake_phrase}'")
                        return True
                        
        except queue.Empty:
            continue
        except Exception as e:
            print(f"⚠️  Error in wake word detection: {e}")
            continue


def _set_permission_from_voice(user_id: int, agree: bool):
    """
    Update the 'permission' flag for the user in the DB.
    """
    db.set_permission(user_id, agree)
    print(
        f"✅ Updated permission for user_id={user_id} "
        f"to {'agree (1)' if agree else 'disagree (2)'}."
    )


def on_yes(user_id: int):
    """What to do when 'yes' is detected."""
    print("✅ Detected YES!")
    _set_permission_from_voice(user_id, True)


def on_no(user_id: int):
    """What to do when 'no' is detected."""
    print("❌ Detected NO!")
    _set_permission_from_voice(user_id, False)


def ask_permission(user_id: int, user_name: str, prompt_path: Path = None):
    """
    Ask the user for privacy permission using audio prompt and voice recognition.
    Plays an audio prompt, then listens for yes/no response within a window.
    Keeps listening until window expires and commits the last detected choice.
    """
    if prompt_path is None:
        prompt_path = PROMPT_AUDIO_PATH
    
    print(f"\n{'='*50}")
    print(f"Privacy Permission Request for: {user_name}")
    print(f"{'='*50}")
    
    if not VOSK_AVAILABLE:
        print("⚠️  Vosk model not available. Cannot process permission request.")
        print("   Please set permission manually or ensure Vosk model is available.")
        return
    
    # Play audio prompt
    print("Playing audio prompt...")
    prompt_played = play_prompt_audio(prompt_path)
    
    if not prompt_played:
        print("⚠️  Could not play audio prompt. Using text prompt instead.")
        print("Please say 'yes' to grant permission or 'no' to deny.")
    
    # Clear the audio queue to remove any old audio from before the prompt
    print("Clearing audio queue...")
    while not audio_q.empty():
        try:
            audio_q.get_nowait()
        except queue.Empty:
            break
    
    # Reset the recognizer to clear its internal state
    if recognizer is not None:
        recognizer.Reset()
    
    # Small delay to ensure we start with fresh audio
    time.sleep(0.2)
    
    prompt_played = play_prompt_audio(BEEP_AUDIO_PATH)
    
    print("Listening for your response...\n")
    
    # Listen for yes/no response within the response window
    start_time = time.time()
    last_detected_choice = None
    
    while time.time() - start_time < RESPONSE_WINDOW_SECONDS:
        try:
            data = audio_q.get(timeout=0.1)
            if recognizer.AcceptWaveform(data):
                result = recognizer.Result()
                result_dict = json.loads(result)
                text = result_dict.get("text", "").lower()
                
                if not text:
                    continue
                
                print(f"Heard: {text}")
                
                # Check for yes/no keywords - keep listening until window expires
                if any(word in text for word in ["yes", "yeah", "yep", "sure", "okay", "ok"]):
                    print("✅ Detected YES!")
                    last_detected_choice = True
                    # Continue listening - don't stop here
                elif any(word in text for word in ["no", "nope", "nah"]):
                    print("❌ Detected NO!")
                    last_detected_choice = False
                    # Continue listening - don't stop here
        except queue.Empty:
            continue
    
    # Response window expired - commit the last detected choice
    elapsed = time.time() - start_time
    if last_detected_choice is not None:
        _set_permission_from_voice(user_id, last_detected_choice)
        print(f"\n✅ Response window expired. Committed last choice: {'YES' if last_detected_choice else 'NO'}")
    else:
        print(f"\n⏱️  Response window expired ({elapsed:.1f} seconds). No response detected.")
        print("   Permission not updated.")


def generate_user_name() -> str:
    """
    Automatically generate a unique user name for new users.
    Uses pattern "User_1", "User_2", etc., checking existing names to avoid conflicts.
    """
    cur = db.con.cursor()
    cur.execute(f"SELECT name FROM {db.USER_TABLE}")
    existing_names = {row[0] for row in cur.fetchall()}
    
    # Find the next available User_X number
    user_num = 1
    while f"User_{user_num}" in existing_names:
        user_num += 1
    
    generated_name = f"User_{user_num}"
    print(f"\n{'='*50}")
    print("New User Detected")
    print(f"{'='*50}")
    print(f"✅ Automatically generated name: {generated_name}")
    return generated_name


def process_user_voice():
    """
    Main pipeline: Record voice, identify or enroll, request permission if new.
    """
    print("\n" + "="*50)
    print("Speaker Recognition Pipeline")
    if RESEMBLYZER_AVAILABLE:
        print("Using Resemblyzer - Say ANY sentence to identify yourself!")
    else:
        print("Using MFCC - Please say a FULL SENTENCE")
    print("="*50)
    if RESEMBLYZER_AVAILABLE:
        print(f"\n✅ You can say ANY sentence - the system identifies WHO is speaking.")
        print(f"   Examples: 'Hello, this is John' or 'I want to access the system'")
        print(f"   The system recognizes your VOICE, not the words you say.")
    else:
        print(f"\n⚠️  Please say a FULL SENTENCE (not just 'yes' or 'no')")
        print(f"   Examples: 'Hello, this is [your name]' or 'I want to access the system'")
        
    
    # play welcome prompt
    play_audio = play_prompt_audio(WELCOME_AUDIO_PATH)
    
    print(f"\nPlease speak for {RECORD_DURATION} seconds...")
    print("Recording in 2 seconds...")
    time.sleep(2)
    
    print("Speak now!\n")
    
    # play beep sound
    play_audio = play_prompt_audio(BEEP_AUDIO_PATH)
    
    # Record voice sample
    audio = record_audio(RECORD_DURATION)
    print("✓ Recording complete. Processing...")
    
    # Try to identify speaker
    match = identify_speaker(audio)
    
    if match:
        # Existing user found
        user_id = match["user_id"]
        user_name = match["name"]
        score = match["score"]
        confidence = match.get("confidence", "unknown")
        
        print(f"\n🎤 Identified speaker: {user_name} (user_id={user_id})")
        if RESEMBLYZER_AVAILABLE:
            print(f"   Similarity score: {score:.4f} (cosine similarity, higher is better)")
        else:
            print(f"   Match score (L2 distance): {score:.4f} (lower is better)")
        print(f"   Confidence: {confidence.upper()}")
        if RESEMBLYZER_AVAILABLE:
            print(f"   ✅ Speaker recognized from your sentence!")
        else:
            print(f"   ⚠️  Note: If you said just 'yes' or 'no', this match may be inaccurate.")
            print(f"      Please say a full sentence for reliable identification.")
        
        # Check permission status (informational only for existing users)
        try:
            permission = db.get_permission(user_id)
            if permission == 1:
                print(f"   ✅ Permission status: Granted")
            elif permission == 2:
                print(f"   ❌ Permission status: Denied")
            else:
                print(f"   ⚠️  Permission status: Not set")
        except ValueError:
            print(f"   ⚠️  Permission status: Unavailable")
        
        print(f"\n✓ Welcome back, {user_name}!")
        
        db.set_last_seen_user(user_id);
        
        # Ask for permission confirmation with audio prompt
        print(f"\n{'='*50}")
        print("Permission Confirmation")
        print(f"{'='*50}")
        print(f"Please confirm your privacy permission preference.")
        ask_permission(user_id, user_name)
    else:
        # New user - need to enroll
        print("\n❌ No matching speaker found in database.")
        print("   This appears to be a first-time user.")
        
        # Generate user name automatically
        user_name = generate_user_name()
        
        # Enroll the user with multiple samples
        print(f"\nEnrolling new user '{user_name}'...")
        print("   We'll record a few more samples to create a reliable voice template.")
        user_id = enroll_new_user(user_name, num_samples=ENROLLMENT_SAMPLES)
        
        # Ask for permission for first-time user
        if user_id:
            db.set_last_seen_user(user_id);
            print(f"\n{'='*50}")
            print("First-time User - Privacy Permission Required")
            print(f"{'='*50}")
            print(f"Welcome, {user_name}! As a first-time user, we need your privacy consent.")
            ask_permission(user_id, user_name)
        else:
            print("⚠️  Could not enroll user or retrieve user_id.")


def main():
    print("="*50)
    print("Voice Recognition & Enrollment Pipeline")
    print("="*50)
    print("\nThis system will:")
    print("  1. Sleep until wake word 'Hi Cursor' is detected")
    print("  2. Record your voice sample (3 seconds)")
    print("  3. Check if your voice matches an existing user in the database")
    print("  4. If new user: enroll you with additional samples, then request privacy permission")
    print("  5. If existing user: welcome you back, then return to sleep mode")
    print("\n(Press Ctrl+C to exit)\n")
    
    if not VOSK_AVAILABLE:
        print("⚠️  WARNING: Vosk model not available. Wake word detection will not work.")
        print("   The system will process users immediately without wake word detection.")
        print("   Please ensure Vosk model is available for full functionality.\n")
    
    try:
        with sd.RawInputStream(
            samplerate=SAMPLE_RATE,
            blocksize=BLOCK_SIZE,
            dtype="int16",
            channels=1,
            callback=audio_callback,
        ):
            while True:
                # Wait for wake word "Hi Cursor"
                if VOSK_AVAILABLE:
                    wake_detected = detect_wake_word("Hi Apple")
                    if not wake_detected:
                        continue  # Keep listening
                
                # Wake word detected or Vosk not available - process user
                print("\n" + "="*50)
                print("🔊 System activated!")
                print("="*50)
                
                process_user_voice()
                
                # After processing, return to sleep mode
                print("\n" + "="*50)
                print("✅ Processing complete. Returning to sleep mode...")
                print("="*50)
                time.sleep(1)  # Brief pause before returning to sleep
                
    except KeyboardInterrupt:
        print("\n\n👋 Exiting voice pipeline.")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        db.close()


if __name__ == "__main__":
    main()

