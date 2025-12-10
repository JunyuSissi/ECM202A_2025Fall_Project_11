from database.face_db import FaceDB
from pathlib import Path
DB_PATH = Path(__file__).resolve().parents[2] / "database" / "face_database.db"
db = FaceDB(str(DB_PATH))
db.clear_all_data(confirm=True)
db.close()
