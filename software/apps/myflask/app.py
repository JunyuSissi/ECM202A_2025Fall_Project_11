"""
Added access to the database

"""

from flask import Flask, render_template, request, redirect, url_for, jsonify
from database.face_db import FaceDB
from pathlib import Path
from face_rec.facial_recognition_updated.facial_recognition_v4 import get_last_detection
import threading
import json

app = Flask(__name__)

DB_PATH = Path(__file__).resolve().parents[2] / "database" / "face_database.db"
db = FaceDB(str(DB_PATH))

current_user_id = None
current_user_lock = threading.Lock()
active_id = None



def get_user_by_id(user_id: int):
    """Helper to fetch a user row as dict, or None."""
    cur = db.con.cursor()
    cur.execute(
        f"SELECT user_id, name, permission, created_at "
        f"FROM {db.USER_TABLE} WHERE user_id = ?",
        (user_id,),
    )
    r = cur.fetchone()
    if not r:
        return None
    return {
        "user_id": r[0],
        "name": r[1],
        "permission": r[2],
        "created_at": r[3],
    }

def _resolve_active_user_id():
    """
    Pick the user ID from the latest detection produced by the facial recognition module.
    """
    det = db.get_last_seen_user()
    return int(det["user_id"]) if det and det.get("user_id") is not None else None

@app.route("/")
def index():
    global active_id
    active_id = _resolve_active_user_id()

    user = get_user_by_id(active_id) if active_id is not None else None

    if active_id is None:
        return (
            "<h1>No active user</h1>"
            "<p>We couldn't determine which user to update. "
            "Make sure the camera has identified you or a user_id is passed.</p>",
            400,
        )
    # Example: render user info if you want
    # return render_template("index.html", user=user)
    
    # Example shows exactly where you’ll later paste iubenda code in the template.
    return render_template("index.html")

@app.route("/access-granted")
def access_granted():
    """
    Mark the active user as having granted access (permission = 2).
    This will be called after the user clicks 'Accept' on the frontend.
    """

    user_id = _resolve_active_user_id()
    

    # Update DB permission to 1 (granted)

    db.set_permission(user_id, True)

    
    return """
    <main style="font:16px system-ui;display:grid;place-items:center;min-height:100vh;">
      <section style="max-width:720px;padding:24px;border:1px solid #e5e7eb;border-radius:14px;">
        <h1>Access Granted ✅</h1>
        <p>Thanks for your choices. You can change them anytime using the floating preferences button.</p>
        <p><a href="/">Back to home</a></p>
      </section>
    </main>
    """

@app.route("/access-denied")
def access_denied():
    """
    Mark the active user as having denied access (permission = 1).
    This will be called after the user clicks 'Reject' on the frontend.
    """

    user_id = _resolve_active_user_id()


    # Update DB permission to 2 (denied)
    db.set_permission(user_id, False)
    
    return """
    <main style="font:16px system-ui;display:grid;place-items:center;min-height:100vh;">
      <section style="max-width:720px;padding:24px;border:1px solid #e5e7eb;border-radius:14px;">
        <h1>Access Limited ❌</h1>
        <p>You chose not to allow non-essential cookies. Some features may be disabled.</p>
        <p>You can change your choice using the floating preferences button.</p>
        <p><a href="/">Back to home</a></p>
      </section>
    </main>
    """

@app.route("/save-consent", methods=["POST"])
def save_consent():
    """
    Save granular consent preferences for the active user.
    Expects JSON body with consent data from iubenda.
    """
    user_id = _resolve_active_user_id()
    if user_id is None:
        return jsonify({"error": "No active user"}), 400
    
    if user_id != active_id:
        return jsonify({"error": "User mismatch"}), 400
    
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No data provided"}), 400
        
        # Save granular consent preferences
        db.set_granular_consent(active_id, data)
        
        # Also update permission based on overall consent
        # According to iubenda docs, preference object structure:
        # {
        #   "id": "xxxxx",
        #   "consent": true/false/undefined,
        #   "purposes": {"1": true/false, "2": true/false, ...},  # GDPR-style
        #   "ccpa": "1YN-",  # US State Law: CCPA string (1=opt-out, Y=yes, N=no, -=N/A)
        #   "uspr": {"s": true, "sh": true, "adv": true},  # US State Law: US Privacy preferences
        #   "tcfv2": {...},  # optional TCF v2 data
        #   "gac": {...}     # optional Google Additional Consent
        # }
        
        has_consent = False
        
        # Check binary consent first
        if data.get("consent") == True:
            has_consent = True
        elif data.get("consent") == False:
            has_consent = False
        # Check US State Law preferences (CCPA, CPRA, VCDPA, etc.)
        elif data.get("uspr"):
            # US Privacy preferences: s (sale), sh (sharing), adv (advertising)
            uspr = data.get("uspr", {})
            # Grant access if user consented to any US Privacy category
            has_consent = uspr.get("s") == True or uspr.get("sh") == True or uspr.get("adv") == True
        elif data.get("ccpa"):
            # CCPA string format: "1YN-" where:
            # - First char "1" = opt-out of sale, otherwise consent given
            # - Second char = sharing consent (Y/N)
            # - Third char = advertising consent (Y/N)
            ccpa_str = str(data.get("ccpa", ""))
            # If doesn't start with "1", user hasn't opted out (consent given)
            has_consent = not ccpa_str.startswith("1")
        # Check GDPR-style purpose-based consent
        elif data.get("purposes"):
            # Granular choice: purposes object contains purpose IDs as keys
            # Values can be boolean (true/false) or numeric (4 = accepted, 0 = rejected)
            # Grant access if at least one purpose is accepted
            purposes = data.get("purposes", {})
            has_consent = any(
                purpose_value == True or purpose_value == 4 or purpose_value == "4"
                for purpose_value in purposes.values()
            )
        
        # Update permission based on determined consent
        db.set_permission(active_id, has_consent)
        
        return jsonify({"success": True, "user_id": active_id})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/status")
def status():
    # Expose the latest detection as JSON
    user_id = _resolve_active_user_id()
    if user_id is None:
        return jsonify({
            "status": "no_face",
            "user_id": None,
            "name": None,
            "permission": 0,
            "granular_consent": None
        })
    
    user = get_user_by_id(user_id)
    if user is None:
        return jsonify({
            "status": "unknown",
            "user_id": None,
            "name": None,
            "permission": 0,
            "granular_consent": None
        })
    
    # Get granular consent if available
    granular_consent = db.get_granular_consent(user_id)
    
    return jsonify({
        "status": "match",
        "user_id": user["user_id"],
        "name": user["name"],
        "permission": int(user["permission"]),  # Return as integer: 0, 1, or 2
        "granular_consent": granular_consent
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)





