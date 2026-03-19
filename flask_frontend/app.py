from flask import Flask, render_template, request, jsonify
import requests
import os

app = Flask(__name__)
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

@app.route("/")
def index():
    return render_template("index.html", backend_url=BACKEND_URL)

@app.route("/analyze", methods=["POST"])
def analyze():
    try:
        resume = request.files.get("resume")
        job_description = request.form.get("job_description", "")
        if not resume or not job_description:
            return jsonify({"error": "Resume and job description are required."}), 400
        response = requests.post(
            f"{BACKEND_URL}/analyze",
            files={"resume": (resume.filename, resume.stream, resume.mimetype)},
            data={"job_description": job_description},
            timeout=180
        )
        return jsonify(response.json()), response.status_code
    except requests.exceptions.ConnectionError:
        return jsonify({"error": "Cannot connect to backend."}), 503
    except requests.exceptions.Timeout:
        return jsonify({"error": "Request timed out."}), 504
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/health")
def health():
    try:
        r = requests.get(f"{BACKEND_URL}/health", timeout=5)
        return jsonify({"backend": r.json(), "frontend": "ok"})
    except:
        return jsonify({"backend": "unreachable", "frontend": "ok"}), 200

@app.route("/eval/metrics")
def eval_metrics():
    try:
        r = requests.get(f"{BACKEND_URL}/eval/metrics", timeout=5)
        return jsonify(r.json())
    except:
        return jsonify({"error": "Unavailable"}), 503

@app.route("/eval/corpus")
def eval_corpus():
    try:
        r = requests.get(f"{BACKEND_URL}/eval/corpus/stats", timeout=5)
        return jsonify(r.json())
    except:
        return jsonify({"error": "Unavailable"}), 503

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)