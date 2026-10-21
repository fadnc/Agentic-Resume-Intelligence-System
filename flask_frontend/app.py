import os
import time
import requests
from flask import Flask, Response, render_template, request, jsonify

app = Flask(__name__)
BACKEND_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:8001")


def _backend_get(path: str, timeout: int = 10):
    return requests.get(f"{BACKEND_URL}{path}", timeout=timeout)


def _backend_post(path: str, **kwargs):
    return requests.post(f"{BACKEND_URL}{path}", **kwargs)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/analyze", methods=["POST"])
def analyze():
    try:
        resume = request.files.get("resume")
        job_description = request.form.get("job_description", "")
        if not resume or not job_description:
            return jsonify({"error": "Resume and job description are required."}), 400

        response = _backend_post(
            "/analyze",
            files={"resume": (resume.filename, resume.stream, resume.mimetype)},
            data={"job_description": job_description},
            timeout=180,
        )
        try:
            payload = response.json()
        except ValueError:
            payload = {
                "error": "Backend returned a non-JSON response.",
                "backend_status": response.status_code,
                "backend_response": response.text[:1000],
            }
        return jsonify(payload), response.status_code

    except requests.exceptions.ConnectionError:
        return jsonify({
            "error": "Backend is starting up (Render free tier cold start). "
                     "Wait 30 seconds and try again."
        }), 503
    except requests.exceptions.Timeout:
        return jsonify({"error": "Analysis timed out. Try again."}), 504
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/recruiter/screen", methods=["POST"])
def recruiter_screen():
    try:
        resumes = request.files.getlist("resumes")
        job_description = request.form.get("job_description", "")
        if not resumes or not job_description:
            return jsonify({"error": "Upload at least one resume and enter a job description."}), 400
        if len(resumes) > 5:
            return jsonify({"error": "Recruiter mode accepts up to five resumes per batch."}), 413
        files = [
            ("resumes", (item.filename, item.stream, item.mimetype or "application/pdf"))
            for item in resumes
        ]
        response = _backend_post(
            "/recruiter/screen", files=files,
            data={"job_description": job_description}, timeout=600,
        )
        try:
            payload = response.json()
        except ValueError:
            payload = {"error": "Backend returned a non-JSON response.", "backend_status": response.status_code}
        return jsonify(payload), response.status_code
    except requests.exceptions.ConnectionError:
        return jsonify({"error": "Backend is unavailable. Start the backend and try again."}), 503
    except requests.exceptions.Timeout:
        return jsonify({"error": "Batch screening timed out. Try a smaller batch."}), 504


@app.route("/export/<format_name>", methods=["POST"])
def export_resume(format_name):
    if format_name not in {"docx", "latex"}:
        return jsonify({"error": "Unsupported export format."}), 404
    try:
        response = _backend_post(
            f"/export/{format_name}", json=request.get_json(silent=True), timeout=20,
        )
        return Response(
            response.content,
            status=response.status_code,
            content_type=response.headers.get("Content-Type", "application/octet-stream"),
            headers={"Content-Disposition": response.headers.get("Content-Disposition", "attachment")},
        )
    except requests.exceptions.ConnectionError:
        return jsonify({"error": "Backend is unavailable. Start the backend and try again."}), 503
    except requests.exceptions.Timeout:
        return jsonify({"error": "Resume export timed out. Try again."}), 504


@app.route("/interview/feedback", methods=["POST"])
def interview_feedback():
    try:
        response = _backend_post(
            "/interview/feedback", json=request.get_json(silent=True), timeout=60,
        )
        try:
            payload = response.json()
        except ValueError:
            payload = {"error": "Backend returned a non-JSON response."}
        return jsonify(payload), response.status_code
    except requests.exceptions.ConnectionError:
        return jsonify({"error": "Backend is unavailable. Start the backend and try again."}), 503
    except requests.exceptions.Timeout:
        return jsonify({"error": "Interview feedback timed out. Try again."}), 504


@app.route("/health")
def health():
    try:
        r = _backend_get("/health", timeout=5)
        return jsonify({"backend": r.json(), "frontend": "ok"})
    except Exception:
        return jsonify({"backend": "starting", "frontend": "ok"}), 200


@app.route("/eval/metrics")
def eval_metrics():
    try:
        r = _backend_get("/eval/metrics", timeout=5)
        return jsonify(r.json())
    except Exception:
        return jsonify({"error": "Unavailable"}), 503


@app.route("/eval/corpus")
def eval_corpus():
    try:
        r = _backend_get("/eval/corpus/stats", timeout=5)
        return jsonify(r.json())
    except Exception:
        return jsonify({"error": "Unavailable"}), 503


if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
