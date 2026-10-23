from pathlib import Path

from flask import (
    Flask,
    render_template,
    request,
    redirect,
    url_for,
    flash,
)
from werkzeug.utils import secure_filename

from movement_analysis import analyze_video


BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"
MODEL_PATH = BASE_DIR / "models" / "pose_landmarker_full.task"

ALLOWED_EXTENSIONS = {"mp4", "mov", "avi", "m4v"}

app = Flask(__name__)
app.secret_key = "change-this-development-secret"

app.config["MAX_CONTENT_LENGTH"] = 150 * 1024 * 1024

UPLOAD_DIR.mkdir(exist_ok=True)
MODEL_PATH.parent.mkdir(exist_ok=True)


def allowed_file(filename):
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS
    )


@app.get("/")
def index():
    return render_template("index.html")


@app.post("/analyze")
def analyze():
    if "video" not in request.files:
        flash("Please choose a video file.")
        return redirect(url_for("index"))

    video = request.files["video"]

    if not video.filename:
        flash("Please choose a video file.")
        return redirect(url_for("index"))

    if not allowed_file(video.filename):
        flash("Upload an MP4, MOV, AVI, or M4V video.")
        return redirect(url_for("index"))

    if not MODEL_PATH.exists():
        flash(
            "The MediaPipe model is missing. "
            "Run python download_model.py first."
        )
        return redirect(url_for("index"))

    filename = secure_filename(video.filename)

    if not filename:
        flash("The filename is invalid.")
        return redirect(url_for("index"))

    # Use a unique temporary filename to avoid upload collisions.
    import uuid

    filename = f"{uuid.uuid4().hex}_{filename}"
    saved_path = UPLOAD_DIR / filename

    try:
        video.save(saved_path)

        result = analyze_video(saved_path, MODEL_PATH)
        result["filename"] = filename

        return render_template(
            "results.html",
            result=result,
        )

    except Exception:
        app.logger.exception("Video analysis failed.")

        flash(
            "We couldn't analyze this video. "
            "Check the video format and try another clip."
        )

        return redirect(url_for("index"))

    finally:
        # Remove the uploaded video after processing.
        try:
            saved_path.unlink(missing_ok=True)
        except OSError:
            app.logger.warning("Could not delete temporary video.")


@app.errorhandler(413)
def file_too_large(error):
    flash("The video is too large. Maximum size is 150 MB.")
    return redirect(url_for("index"))


if __name__ == "__main__":
    # Development server only. Do not expose this server publicly.
    app.run(debug=True)
