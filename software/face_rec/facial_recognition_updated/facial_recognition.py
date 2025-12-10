"""
Pi-side facial recognition loop that reads embeddings from SQLite (via FaceDB)
and publishes the latest recognized user into a small cache table for the web
app to consume. No local pickle, no GPIO.
"""

import time
import sys
from pathlib import Path
from typing import Dict, Optional
from uuid import uuid4

import cv2
import face_recognition
import numpy as np
from picamera2 import Picamera2

from database.face_db import FaceDB

# Add project root to path for TTS import
project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

try:
    from voice_env.tts_helper import speak_user_instructions
    TTS_AVAILABLE = True
except ImportError:
    print("Warning: TTS helper not available. Speech will be disabled.")
    TTS_AVAILABLE = False
    def speak_user_instructions(name=None):
        pass

# Temporary switch to silence voice prompts without touching the helper.
DISABLE_TTS_PROMPTS = True

DB_PATH = Path(__file__).resolve().parents[2] / "database" / "face_database.db"
db = FaceDB(str(DB_PATH))  # Loads face encodings + permissions from SQLite

# --- DB matching/enrollment settings (tune per model/environment) ---
MODEL_TAG = "dlib_128"   # embedding model tag used when enrolling faces in SQLite
METRIC = "l2"            # distance metric for match: 'l2' (lower=better) or 'cos' (higher=better)
THRESHOLD = 0.38         # match threshold; lower is stricter for l2, higher is stricter for cos
AUTO_ENROLL = True       # automatically enroll truly new faces (see clustering logic below)
MAX_FACES_PER_USER = 8   # cap how many face vectors we store per user to avoid runaway inserts

# Camera will be initialized only when running as a script (avoids grabbing device on import)
picam2: Optional[Picamera2] = None

# Globals used by draw_results()
cv_scaler = 4  # downscale factor for faster processing (must be an integer)
face_locations = []
face_encodings = []
face_names = []
face_perms = []
frame_count = 0
start_time = time.time()
fps = 0
last_detection: Optional[Dict] = None  # in-memory handoff for the web app

# --- Short-term unknown-face clustering for auto-enroll ---
# We don't want to create a new user_id every frame while the same person is
# moving their head. Instead, we cluster several consecutive "unknown" encodings
# that are close together, then create ONE Visitor user and enroll a few samples.
PENDING_MAX_ENCODINGS = 5        # number of similar frames to collect before committing
PENDING_MAX_SECONDS = 3.0        # max time window for a cluster
PENDING_CLUSTER_TOL = 0.45       # L2 distance threshold to keep encodings in same cluster

pending_cluster: Dict = {
    "encodings": [],
    "user_id": None,
    "name": None,
    "started_at": 0.0,
}


def _update_pending_cluster_for_auto_enroll(face_encoding: np.ndarray):
    """
    Aggregate several similar 'unknown' encodings into a short-lived cluster.
    Once we have enough consistent samples, create a single Visitor-XXXX user
    and enroll those samples under one user_id.

    Returns:
        dict | None with keys {user_id, face_id, name} when a user is (or was)
        committed, otherwise None while still collecting samples.
    """
    global pending_cluster
    now = time.time()

    enc = np.asarray(face_encoding, dtype=np.float32).ravel()
    cluster_encs = pending_cluster["encodings"]

    # Start a new cluster if empty or too old
    if not cluster_encs or (now - pending_cluster["started_at"]) > PENDING_MAX_SECONDS:
        pending_cluster = {
            "encodings": [enc],
            "user_id": None,
            "name": None,
            "started_at": now,
        }
        return None

    # Compare to mean encoding of current cluster
    mean_enc = np.mean(np.stack(cluster_encs, axis=0), axis=0)
    d = float(np.linalg.norm(mean_enc - enc))
    if d > PENDING_CLUSTER_TOL:
        # Too far from current cluster → treat as a new person, reset cluster
        pending_cluster = {
            "encodings": [enc],
            "user_id": None,
            "name": None,
            "started_at": now,
        }
        return None

    # Same person within cluster tolerance
    cluster_encs.append(enc)

    # If we already have a committed user_id, just reuse it
    if pending_cluster["user_id"] is not None:
        return {
            "user_id": pending_cluster["user_id"],
            "face_id": None,
            "name": pending_cluster["name"],
        }

    # Not enough samples yet → keep collecting
    if len(cluster_encs) < PENDING_MAX_ENCODINGS:
        return None

    # Commit this cluster as a new Visitor user
    visitor_name = f"Visitor-{uuid4().hex[:8]}"
    user_id = db.create_user(visitor_name, permission=False)
    last_face_id = None
    for enc_i in cluster_encs[:MAX_FACES_PER_USER]:
        last_face_id = db.enroll(
            name=visitor_name,
            encoding=enc_i,
            img_path=None,
            model_tag=MODEL_TAG,
            permission=False,
        )

    pending_cluster["user_id"] = user_id
    pending_cluster["name"] = visitor_name

    return {"user_id": user_id, "face_id": last_face_id, "name": visitor_name}


def process_frame(frame):
    """
    Run detection + DB lookup. Saves the first match in memory for a GET hook
    (the web app can read it via get_last_detection()).
    """
    global face_locations, face_encodings, face_names, face_perms, last_detection

    resized_frame = cv2.resize(frame, (0, 0), fx=(1 / cv_scaler), fy=(1 / cv_scaler))
    rgb_resized_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)

    face_locations = face_recognition.face_locations(rgb_resized_frame)
    face_encodings = face_recognition.face_encodings(rgb_resized_frame, face_locations, model="large")

    face_names = []
    face_perms = []
    first_match: Optional[Dict] = None
    unknown_info: Optional[Dict] = None  # captures auto-enrolled unknown IDs

    # We use the total number of faces in this frame to decide if auto-enrollment is allowed.
    num_faces = len(face_encodings)

    for face_encoding in face_encodings:
        match = db.find_match(
            query_encoding=face_encoding,
            model_tag=MODEL_TAG,
            metric=METRIC,
            threshold=THRESHOLD,
        )

        if match:
            # Existing user: only read identity. Do NOT keep auto-adding encodings here,
            # because a wrong first match could get reinforced with many extra samples.
            name = match["name"] or f"User#{match['user_id']}"
            perm = bool(match["permission"])
            if first_match is None:
                first_match = match
        else:
            name = "Unknown"
            perm = False
            # Only consider auto-enroll when *exactly one* face is in view.
            # Multiple faces → treat as unknown but do not write to DB.
            if AUTO_ENROLL and num_faces == 1:
                cluster_result = _update_pending_cluster_for_auto_enroll(face_encoding)
                if cluster_result:
                    unknown_info = cluster_result
                    name = cluster_result["name"]

        face_names.append(name)
        face_perms.append(perm)

    # Keep a simple in-memory record the web app can read via get_last_detection()
    timestamp = time.time()
    if first_match:
        last_detection = {
            "status": "match",
            "user_id": first_match.get("user_id"),
            "face_id": first_match.get("face_id"),
            "name": first_match.get("name"),
            "permission": bool(first_match.get("permission")),
            "score": first_match.get("score"),
            "seen_at": timestamp,
        }
    elif face_encodings:
        # At least one face found but no match; include auto-enrolled ID if present.
        last_detection = {
            "status": "unknown",
            "user_id": unknown_info["user_id"] if unknown_info else None,
            "face_id": unknown_info["face_id"] if unknown_info else None,
            "name": unknown_info["name"] if unknown_info else None,
            "permission": False,
            "score": None,
            "seen_at": timestamp,
        }
    else:
        last_detection = {"status": "no_face", "user_id": None, "seen_at": timestamp}

    return frame


def get_last_detection() -> Optional[Dict]:
    """
    Returns dict with keys: status, user_id, face_id, name, permission, score, seen_at
    or None if nothing has been processed yet.
    """
    # Return a shallow copy so callers don't accidentally mutate the shared state.
    
    return dict(last_detection) if last_detection else None


def draw_results(frame):
    """Draw face boxes, names, and authorization status onto the frame."""
    for (top, right, bottom, left), name, perm in zip(face_locations, face_names, face_perms):
        top *= cv_scaler
        right *= cv_scaler
        bottom *= cv_scaler
        left *= cv_scaler

        cv2.rectangle(frame, (left, top), (right, bottom), (244, 42, 3), 3)
        cv2.rectangle(frame, (left - 3, top - 35), (right + 3, top), (244, 42, 3), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, top - 6), font, 1.0, (255, 255, 255), 1)

        if perm:
            cv2.putText(frame, "Authorized", (left + 6, bottom + 23), font, 0.6, (0, 255, 0), 1)

    return frame


def calculate_fps():
    """Compute a rolling FPS based on processed frames and elapsed time."""
    global frame_count, start_time, fps
    frame_count += 1
    elapsed_time = time.time() - start_time
    if elapsed_time > 1:
        fps = frame_count / elapsed_time
        frame_count = 0
        start_time = time.time()
    return fps


if __name__ == "__main__":
    picam2 = Picamera2()
    picam2.configure(picam2.create_preview_configuration(main={"format": "XRGB8888", "size": (1280, 720)}))
    picam2.start()

    last_last_detection = {"status": "no_face", "user_id": None, "seen_at": time.time()}
    last_spoken_user_id = None  # Track which user we've already spoken to
    try:
        while True:
            frame = picam2.capture_array()

            processed_frame = process_frame(frame)
            # Only update last_seen when we actually have a detection with a user_id
            if last_detection and last_detection.get("user_id") != last_last_detection.get("user_id"):
                if last_detection.get("user_id") is not None:
                    db.set_last_seen_user(last_detection["user_id"])
                    
                    # Speak instructions when a new user appears (only once per user session)
                    if not DISABLE_TTS_PROMPTS and last_detection.get("user_id") != last_spoken_user_id:
                        user_name = last_detection.get("name")
                        # Only speak if we have a name (not "Unknown")
                        if user_name and user_name != "Unknown":
                            speak_user_instructions(user_name)
                        else:
                            speak_user_instructions()
                        last_spoken_user_id = last_detection.get("user_id")
                
                # store a shallow copy so later mutations don't affect the comparison
                last_last_detection = dict(last_detection)
            elif last_detection and last_detection.get("user_id") is None:
                # No face detected, reset spoken user tracking
                last_spoken_user_id = None

            display_frame = draw_results(processed_frame)

            current_fps = calculate_fps()
            cv2.putText(display_frame, f"FPS: {current_fps:.1f}", (display_frame.shape[1] - 150, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            cv2.imshow("Video", display_frame)

            if cv2.waitKey(1) == ord("q"):
                break
    finally:
        cv2.destroyAllWindows()
        picam2.stop()
        db.close()


