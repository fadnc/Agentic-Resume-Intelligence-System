import os
import time
import requests
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")


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
        return jsonify(response.json()), response.status_code

    except requests.exceptions.ConnectionError:
        return jsonify({
            "error": "Backend is starting up (Render free tier cold start). "
                     "Wait 30 seconds and try again."
        }), 503
    except requests.exceptions.Timeout:
        return jsonify({"error": "Analysis timed out. Try again."}), 504
    except Exception as e:
        return jsonify({"error": str(e)}), 500


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