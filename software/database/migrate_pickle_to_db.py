"""
This file load the encoding from encodings.pickle into the SQLite database
"""

# migrate_pickle_to_db.py
import pickle, os
import numpy as np
from face_db import FaceDB

DB_PATH   = "face_database.db"
PICKLE    = "encodings.pickle"
MODEL_TAG = "dlib_128"   # face_recognition 128-D embedding
DEFAULT_PERMISSION = False  # set per your policy; or add a name->permission map below

# Optional allow-list to set permission=True for certain names
PERM_ALLOWLIST = {
    # "alice": True,
    # "john": True,
}

def main():
    if not os.path.exists(PICKLE):
        raise FileNotFoundError(f"{PICKLE} not found")

    print("[INFO] loading encodings.pickle ...")
    with open(PICKLE, "rb") as f:
        data = pickle.loads(f.read())
    encs  = data["encodings"]  # list of 128-d vectors (float64)
    names = data["names"]      # list of strings

    if len(encs) != len(names):
        raise RuntimeError("encodings and names lengths differ")

    db = FaceDB(DB_PATH)
    inserted = 0
    try:
        for enc, name in zip(encs, names):
            # cast to float32 to store compactly
            enc = np.asarray(enc, dtype=np.float32)
            perm = PERM_ALLOWLIST.get(name, DEFAULT_PERMISSION)
            db.enroll(name=name, encoding=enc, img_path=None, model_tag=MODEL_TAG, permission=perm)
            inserted += 1
        print(f"[OK] migrated {inserted} encodings into DB ({DB_PATH}) with model_tag='{MODEL_TAG}'")
    finally:
        db.close()

if __name__ == "__main__":
    main()

