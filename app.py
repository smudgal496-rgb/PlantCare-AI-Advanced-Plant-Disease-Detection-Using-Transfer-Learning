import os
import secrets
from datetime import timedelta

from flask import Flask, jsonify, redirect, render_template, request, send_from_directory, session, url_for
from PIL import Image
from werkzeug.utils import secure_filename


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "_uploads")
STATIC_FOLDER = os.path.join(BASE_DIR, "static")
STATIC_IMAGES_FOLDER = os.path.join(STATIC_FOLDER, "images")

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "webp"}


def ensure_dirs() -> None:
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    os.makedirs(STATIC_IMAGES_FOLDER, exist_ok=True)


def allowed_file(filename: str) -> bool:
    if "." not in filename:
        return False
    ext = filename.rsplit(".", 1)[1].lower()
    return ext in ALLOWED_EXTENSIONS


# ----------------------------
# Model: load and predict (implement in model_loader.py)
# ----------------------------
from model_loader import load_model, predict_image


app = Flask(__name__, static_folder="static", template_folder="templates")
app.secret_key = os.environ.get("FLASK_SECRET_KEY", secrets.token_hex(16))
app.permanent_session_lifetime = timedelta(hours=2)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["STATIC_FOLDER"] = STATIC_FOLDER


# ----------------------------
# Static pages (your current frontend)
# ----------------------------
@app.get("/")
def home():
    return send_from_directory(BASE_DIR, "index.html")

@app.get("/index.html")
def home_html():
    return send_from_directory(BASE_DIR, "index.html")


@app.get("/about")
def about():
    return send_from_directory(BASE_DIR, "about.html")

@app.get("/about.html")
def about_html():
    return send_from_directory(BASE_DIR, "about.html")


@app.get("/upload")
def upload():
    return send_from_directory(BASE_DIR, "upload.html")

@app.get("/upload.html")
def upload_html():
    return send_from_directory(BASE_DIR, "upload.html")


@app.get("/result-page")
def result_page():
    # Frontend JS uses sessionStorage; backend /result route below is for server-rendered fallback.
    return send_from_directory(BASE_DIR, "result.html")

@app.get("/result.html")
def result_html():
    return send_from_directory(BASE_DIR, "result.html")


@app.get("/styles.css")
def styles():
    return send_from_directory(BASE_DIR, "styles.css")


# ----------------------------
# API: predict
# ----------------------------
@app.post("/predict")
def predict():
    ensure_dirs()

    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    if not file or file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    if not allowed_file(file.filename):
        return jsonify({"error": "Unsupported file type. Please upload JPG/PNG/WEBP."}), 400

    filename = secure_filename(file.filename)
    token = secrets.token_hex(8)
    temp_name = f"temp_{token}_{filename}"
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], temp_name)
    file.save(filepath)

    forced_mode = request.form.get("mode")  # optional: 'healthy'/'disease' for demo

    try:
        prediction = predict_image(filepath, forced_mode=forced_mode)

        # Save to static/images for browser display
        static_filename = f"upload_{token}.jpg"
        static_disk_path = os.path.join(STATIC_IMAGES_FOLDER, static_filename)
        Image.open(filepath).convert("RGB").save(static_disk_path, quality=92)

        # Store in session for /result server route
        session.permanent = True
        session["prediction"] = prediction
        session["image_path"] = f"images/{static_filename}"

        # Cleanup temp file
        try:
            os.remove(filepath)
        except OSError:
            pass

        return jsonify(
            {
                "success": True,
                "prediction": prediction,
                "image_url": url_for("static", filename=session["image_path"]),
            }
        )
    except Exception as e:
        try:
            if os.path.exists(filepath):
                os.remove(filepath)
        except OSError:
            pass
        return jsonify({"error": str(e)}), 500


# ----------------------------
# Server-rendered result (optional)
# ----------------------------
@app.get("/result")
def result():
    prediction = session.get("prediction")
    image_path = session.get("image_path")
    if not prediction or not image_path:
        return redirect(url_for("upload"))
    return render_template("result_server.html", prediction=prediction, image_url=url_for("static", filename=image_path))


if __name__ == "__main__":
    ensure_dirs()
    load_model()
    app.run(debug=True, host="0.0.0.0", port=5000)

