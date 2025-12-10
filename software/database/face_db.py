# face_db.py
"""
This file contains the FaceDB class that add and check the content of the database containing the user information
"""

import sqlite3
from typing import Optional, Tuple, List, Dict
import numpy as np
import threading

class FaceDB:
    # === CHANGED: table/column names to match DB ===
    USER_TABLE = "users"
    FACE_TABLE = "user_faces"
    FACE_VEC_COL = "encoding"      # enamed to 'encoding'
    VOICE_TABLE = "user_voices"
    VOICE_VEC_COL = "encoding"      # voice embedding encoding
    # ================================================

    def __init__(self, db_path: str = "face_database.db"):
        self.con = sqlite3.connect(db_path, check_same_thread=False)
        self.con.execute("PRAGMA foreign_keys=ON;")
        self.con.execute("PRAGMA synchronous=NORMAL;")
        self.con.execute("PRAGMA journal_mode=WAL;")

        # Ensure permission column exists (auto-migrate if needed)
        self._ensure_permission_column()
        self._ensure_last_seen_column()
        self._ensure_granular_consent_column()
        
        # Ensure user_voices table exists
        self._ensure_voice_table()

        # Helpful indexes (safe if they already exist)
        self.con.execute(f"CREATE INDEX IF NOT EXISTS idx_{self.USER_TABLE}_name ON {self.USER_TABLE}(name);")
        self.con.execute(f"CREATE INDEX IF NOT EXISTS idx_{self.FACE_TABLE}_user_id ON {self.FACE_TABLE}(user_id);")
        self.con.execute(f"CREATE INDEX IF NOT EXISTS idx_{self.VOICE_TABLE}_user_id ON {self.VOICE_TABLE}(user_id);")
        self.con.commit()

    # --- utility: ensure users.permission exists ---
    def _ensure_permission_column(self):
        cur = self.con.cursor()
        cur.execute(f"PRAGMA table_info({self.USER_TABLE});")
        cols = {row[1] for row in cur.fetchall()}  # row[1] is column name
        if "permission" not in cols:
            # add boolean-like column (0/1)
            self.con.execute(f"ALTER TABLE {self.USER_TABLE} ADD COLUMN permission INTEGER NOT NULL DEFAULT 0;")
            self.con.commit()

    # ---------- helpers ----------
    @staticmethod
    def _enc_to_blob(encoding: np.ndarray) -> bytes:
        return np.asarray(encoding, dtype=np.float32).ravel().tobytes()

    @staticmethod
    def _blob_to_enc(blob: bytes, dim: int) -> np.ndarray:
        arr = np.frombuffer(blob, dtype=np.float32)
        if dim > 0 and arr.size != dim:
            raise ValueError(f"Expected {dim} floats, got {arr.size}")
        return arr

    @staticmethod
    def _l2(a: np.ndarray, b: np.ndarray) -> float:
        a = np.asarray(a, dtype=np.float32).ravel()
        b = np.asarray(b, dtype=np.float32).ravel()
        return float(np.linalg.norm(a - b))

    @staticmethod
    def _cos(a: np.ndarray, b: np.ndarray) -> float:
        a = np.asarray(a, dtype=np.float32).ravel()
        b = np.asarray(b, dtype=np.float32).ravel()
        denom = (np.linalg.norm(a) * np.linalg.norm(b))
        return 0.0 if denom == 0 else float(np.dot(a, b) / denom)

    # ---------- user CRUD (with permission) ----------
    def get_user_by_name(self, name: str) -> Optional[Dict]:
        cur = self.con.cursor()
        cur.execute(
            f"SELECT user_id, name, permission, created_at FROM {self.USER_TABLE} WHERE name = ?",
            (name,)
        )
        r = cur.fetchone()
        if not r:
            return None
        return {"user_id": r[0], "name": r[1], "permission": int(r[2]), "created_at": r[3]}

    def create_user(self, name: str, permission: bool = False) -> int:
        cur = self.con.cursor()
        cur.execute(
            f"INSERT INTO {self.USER_TABLE}(name, permission) VALUES (?, ?)",
            (name, 0)
        )
        self.con.commit()
        return int(cur.lastrowid)

    def get_or_create_user(self, name: str, permission: bool = False) -> int:
        u = self.get_user_by_name(name)
        if u:
            return int(u["user_id"])
        return self.create_user(name, permission=permission)

    def set_permission(self, user_id: int, permission: bool) -> None:
        cur = self.con.cursor()
        cur.execute(
            f"UPDATE {self.USER_TABLE} SET permission = ? WHERE user_id = ?",
            (1 if permission else 2, user_id)
        )
        self.con.commit()

    def get_permission(self, user_id: int) -> bool:
        cur = self.con.cursor()
        cur.execute(
            f"SELECT permission FROM {self.USER_TABLE} WHERE user_id = ?",
            (user_id,)
        )
        r = cur.fetchone()
        if not r:
            raise ValueError(f"user_id {user_id} not found")
        return int(r[0])

    def set_granular_consent(self, user_id: int, consent_data: Dict) -> None:
        """
        Save granular consent preferences for a user.
        consent_data should be a dict with purposes, vendors, etc.
        """
        import json
        cur = self.con.cursor()
        consent_json = json.dumps(consent_data)
        cur.execute(
            f"UPDATE {self.USER_TABLE} SET granular_consent = ? WHERE user_id = ?",
            (consent_json, user_id)
        )
        self.con.commit()

    def get_granular_consent(self, user_id: int) -> Optional[Dict]:
        """
        Retrieve granular consent preferences for a user.
        Returns None if no consent data is stored.
        """
        import json
        cur = self.con.cursor()
        cur.execute(
            f"SELECT granular_consent FROM {self.USER_TABLE} WHERE user_id = ?",
            (user_id,)
        )
        r = cur.fetchone()
        if not r or r[0] is None:
            return None
        try:
            return json.loads(r[0])
        except (json.JSONDecodeError, TypeError):
            return None
        
  # --- NEW: ensure users.last_seen exists ---
    def _ensure_last_seen_column(self):
        cur = self.con.cursor()
        cur.execute(f"PRAGMA table_info({self.USER_TABLE});")
        cols = {row[1] for row in cur.fetchall()}
        if "last_seen" not in cols:
            self.con.execute(
                f"ALTER TABLE {self.USER_TABLE} "
                f"ADD COLUMN last_seen INTEGER NOT NULL DEFAULT 0;"
            )
            self.con.commit()

    # --- NEW: ensure users.granular_consent exists ---
    def _ensure_granular_consent_column(self):
        cur = self.con.cursor()
        cur.execute(f"PRAGMA table_info({self.USER_TABLE});")
        cols = {row[1] for row in cur.fetchall()}
        if "granular_consent" not in cols:
            self.con.execute(
                f"ALTER TABLE {self.USER_TABLE} "
                f"ADD COLUMN granular_consent TEXT;"
            )
            self.con.commit()

    # --- NEW: ensure user_voices table exists ---
    def _ensure_voice_table(self):
        """Create the user_voices table if it doesn't exist."""
        cur = self.con.cursor()
        cur.execute(
            f"SELECT name FROM sqlite_master WHERE type='table' AND name='{self.VOICE_TABLE}';"
        )
        if not cur.fetchone():
            # Create the table with only user_id and encoding BLOB
            self.con.execute(f"""
                CREATE TABLE {self.VOICE_TABLE} (
                    voice_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL,
                    {self.VOICE_VEC_COL} BLOB NOT NULL,
                    FOREIGN KEY (user_id) REFERENCES {self.USER_TABLE}(user_id) ON DELETE CASCADE
                );
            """)
            self.con.commit()

    # ---------- NEW helpers for "last person appeared" ----------
    def set_last_seen_user(self, user_id: int) -> None:
        """
        Mark exactly one user as the last seen person.
        All other users get last_seen = 0, this one gets last_seen = 1.
        """
        cur = self.con.cursor()
        # Clear previous flags
        cur.execute(f"UPDATE {self.USER_TABLE} SET last_seen = 0;")
        # Set current one
        cur.execute(
            f"UPDATE {self.USER_TABLE} SET last_seen = 1 WHERE user_id = ?;",
            (user_id,),
        )
        self.con.commit()

    def get_last_seen_user(self):
        """
        Return the last-seen user as a dict, or None if none is set.
        """
        cur = self.con.cursor()
        cur.execute(
            f"SELECT user_id, name, permission, created_at "
            f"FROM {self.USER_TABLE} "
            f"WHERE last_seen = 1 "
            f"ORDER BY created_at DESC "
            f"LIMIT 1;"
        )
        r = cur.fetchone()
        if not r:
            return None
        return {
            "user_id": r[0],
            "name": r[1],
            "permission": bool(r[2]),
            "created_at": r[3],
        }

    # ---------- faces ----------
    def enroll(self, *, name: str, encoding: np.ndarray, img_path: str | None, model_tag: str,
               permission: bool = False) -> int:
        user_id = self.get_or_create_user(name, permission=permission)
        dim = int(np.asarray(encoding).size)
        cur = self.con.cursor()
        cur.execute(
            f"INSERT INTO {self.FACE_TABLE}(user_id, embed_dim, {self.FACE_VEC_COL}, img_path, model_tag) "
            f"VALUES (?, ?, ?, ?, ?)",
            (user_id, dim, self._enc_to_blob(encoding), img_path, model_tag)
        )
        self.con.commit()
        return int(cur.lastrowid)

    def list_user_faces(self, user_id: int) -> List[Dict]:
        cur = self.con.cursor()
        cur.execute(
            f"SELECT face_id, embed_dim, {self.FACE_VEC_COL}, img_path, model_tag, created_at "
            f"FROM {self.FACE_TABLE} WHERE user_id=?",
            (user_id,)
        )
        out = []
        for face_id, dim, blob, img_path, model_tag, created_at in cur.fetchall():
            out.append({
                "face_id": int(face_id),
                "encoding": self._blob_to_enc(blob, dim),
                "embed_dim": int(dim),
                "img_path": img_path,
                "model_tag": model_tag,
                "created_at": created_at
            })
        return out

    def _iter_candidates(self, model_tag: Optional[str]):
        cur = self.con.cursor()
        if model_tag:
            cur.execute(
                f"SELECT face_id, user_id, embed_dim, {self.FACE_VEC_COL} FROM {self.FACE_TABLE} WHERE model_tag=?",
                (model_tag,)
            )
        else:
            cur.execute(
                f"SELECT face_id, user_id, embed_dim, {self.FACE_VEC_COL} FROM {self.FACE_TABLE}"
            )
        for face_id, user_id, dim, blob in cur.fetchall():
            yield int(face_id), int(user_id), self._blob_to_enc(blob, dim)
    
    def count_faces(self, user_id: int, model_tag: Optional[str] = None) -> int:
        """Return how many face vectors a user has stored (optionally filtered by model tag)."""
        cur = self.con.cursor()
        if model_tag:
            cur.execute(
                f"SELECT COUNT(*) FROM {self.FACE_TABLE} WHERE user_id=? AND model_tag=?",
                (user_id, model_tag),
            )
        else:
            cur.execute(
                f"SELECT COUNT(*) FROM {self.FACE_TABLE} WHERE user_id=?",
                (user_id,),
            )
        row = cur.fetchone()
        return int(row[0]) if row else 0
    
    def find_match(
        self,
        *,
        query_encoding: np.ndarray,
        model_tag: Optional[str],
        metric: str = "l2",
        threshold: float = 0.6
    ) -> Optional[Dict]:
        best = None
        if metric.lower() == "l2":
            for face_id, user_id, enc in self._iter_candidates(model_tag):
                d = self._l2(query_encoding, enc)
                if d <= threshold and (best is None or d < best["score"]):
                    best = {"user_id": user_id, "face_id": face_id, "score": d}
        elif metric.lower() in ("cos", "cosine"):
            for face_id, user_id, enc in self._iter_candidates(model_tag):
                s = self._cos(query_encoding, enc)
                if s >= threshold and (best is None or s > best["score"]):
                    best = {"user_id": user_id, "face_id": face_id, "score": s}
        else:
            raise ValueError("metric must be 'l2' or 'cos'")

        if best is None:
            return None

        # Attach name & permission
        cur = self.con.cursor()
        cur.execute(
            f"SELECT name, permission FROM {self.USER_TABLE} WHERE user_id = ?",
            (best["user_id"],)
        )
        r = cur.fetchone()
        best["name"] = r[0] if r else None
        best["permission"] = bool(int(r[1])) if r else False
        return best

    # ---------- voices ----------
    def enroll_voice(self, *, name: str, encoding: np.ndarray, permission: bool = False) -> int:
        """Enroll a voice embedding for a user."""
        user_id = self.get_or_create_user(name, permission=permission)
        cur = self.con.cursor()
        cur.execute(
            f"INSERT INTO {self.VOICE_TABLE}(user_id, {self.VOICE_VEC_COL}) "
            f"VALUES (?, ?)",
            (user_id, self._enc_to_blob(encoding))
        )
        self.con.commit()
        return int(cur.lastrowid)

    def list_user_voices(self, user_id: int, embed_dim: int = 0) -> List[Dict]:
        """List all voice embeddings for a user.
        
        Parameters
        ----------
        user_id : int
            The user ID to list voices for
        embed_dim : int
            The expected embedding dimension. If 0, will try to infer from the BLOB size.
        """
        cur = self.con.cursor()
        cur.execute(
            f"SELECT voice_id, {self.VOICE_VEC_COL} "
            f"FROM {self.VOICE_TABLE} WHERE user_id=?",
            (user_id,)
        )
        out = []
        for voice_id, blob in cur.fetchall():
            # If embed_dim not provided, infer from blob size (assuming float32)
            if embed_dim == 0:
                dim = len(blob) // 4  # float32 is 4 bytes
            else:
                dim = embed_dim
            out.append({
                "voice_id": int(voice_id),
                "encoding": self._blob_to_enc(blob, dim)
            })
        return out

    def _iter_voice_candidates(self, embed_dim: int = 0):
        """Iterator over voice embeddings for matching.
        
        Parameters
        ----------
        embed_dim : int
            The expected embedding dimension. If 0, will try to infer from the BLOB size.
        """
        cur = self.con.cursor()
        cur.execute(
            f"SELECT voice_id, user_id, {self.VOICE_VEC_COL} FROM {self.VOICE_TABLE}"
        )
        for voice_id, user_id, blob in cur.fetchall():
            # If embed_dim not provided, infer from blob size (assuming float32)
            if embed_dim == 0:
                dim = len(blob) // 4  # float32 is 4 bytes
            else:
                dim = embed_dim
            yield int(voice_id), int(user_id), self._blob_to_enc(blob, dim)
    
    def count_voices(self, user_id: int) -> int:
        """Return how many voice vectors a user has stored."""
        cur = self.con.cursor()
        cur.execute(
            f"SELECT COUNT(*) FROM {self.VOICE_TABLE} WHERE user_id=?",
            (user_id,),
        )
        row = cur.fetchone()
        return int(row[0]) if row else 0
    
    def find_voice_match(
        self,
        *,
        query_encoding: np.ndarray,
        embed_dim: int = 0,
        metric: str = "l2",
        threshold: float = 0.6
    ) -> Optional[Dict]:
        """Find the best matching voice embedding.
        
        Parameters
        ----------
        query_encoding : np.ndarray
            The voice embedding to match against
        embed_dim : int
            The expected embedding dimension. If 0, will infer from query_encoding size.
        metric : str
            Distance metric to use: 'l2' or 'cos' (cosine)
        threshold : float
            Threshold for matching
        """
        if embed_dim == 0:
            embed_dim = int(np.asarray(query_encoding).size)
        
        best = None
        if metric.lower() == "l2":
            for voice_id, user_id, enc in self._iter_voice_candidates(embed_dim):
                d = self._l2(query_encoding, enc)
                if d <= threshold and (best is None or d < best["score"]):
                    best = {"user_id": user_id, "voice_id": voice_id, "score": d}
        elif metric.lower() in ("cos", "cosine"):
            for voice_id, user_id, enc in self._iter_voice_candidates(embed_dim):
                s = self._cos(query_encoding, enc)
                if s >= threshold and (best is None or s > best["score"]):
                    best = {"user_id": user_id, "voice_id": voice_id, "score": s}
        else:
            raise ValueError("metric must be 'l2' or 'cos'")

        if best is None:
            return None

        # Attach name & permission
        cur = self.con.cursor()
        cur.execute(
            f"SELECT name, permission FROM {self.USER_TABLE} WHERE user_id = ?",
            (best["user_id"],)
        )
        r = cur.fetchone()
        best["name"] = r[0] if r else None
        best["permission"] = bool(int(r[1])) if r else False
        return best

    def close(self):
        try:
            self.con.close()
        except Exception:
            pass

    def clear_all_data(self, confirm: bool = False) -> None:
        """
        Delete all rows from the users and user_faces tables.
        Does NOT delete the database file itself.

        Parameters
        ----------
        confirm : bool
            Must be True to actually perform deletion. Prevents accidental wipes.
        """
        if not confirm:
            raise ValueError("Refusing to delete data. Call with confirm=True to proceed.")

        cur = self.con.cursor()
        # Disable foreign key checks temporarily to avoid constraint errors
        cur.execute("PRAGMA foreign_keys = OFF;")
        self.con.commit()

        try:
            # Delete all rows
            cur.execute(f"DELETE FROM {self.FACE_TABLE};")
            cur.execute(f"DELETE FROM {self.VOICE_TABLE};")
            cur.execute(f"DELETE FROM {self.USER_TABLE};")
            self.con.commit()
            print("[INFO] All database content deleted successfully.")
        finally:
            cur.execute("PRAGMA foreign_keys = ON;")
            self.con.commit()


