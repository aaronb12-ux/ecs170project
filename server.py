from flask import Flask, request, jsonify
from flask_cors import CORS
#from backend.main import run_baseline_overlap, run_bert, run_w2v, run_bart
from baseline.tf_idf.baseline_tf_idf import run_baseline_tfidf


app = Flask(__name__)
CORS(app, origins="*", supports_credentials=True)

@app.route("/api/analyze", methods=["POST"])
def analyze():
    data = request.get_json(silent=True) or {}
    resume = data.get("resume")
    job = data.get("job")

    if resume is None:
        return jsonify({"error": "resume missing"}), 400
    if job is None:
        return jsonify({"error": "job missing"}), 400

    # run all models
    baseline_result = run_baseline_tfidf(resume, job)
    #bert_result = run_bert(resume, job)
    #w2v_result = run_w2v(resume, job)
    #bart_result = run_bart(resume, job)

    result = baseline_result #Can change/comment out model calls depending on which to run

    response = {
        "score": result["similarityScore"],
        "strengths": result["strengths"],
        "weaknesses": result["weaknesses"]
    }

    return jsonify(response), 200

@app.route("/", methods=["GET"])
def index():
    return "Resume Analyzer API (local). POST /api/analyze with JSON {resume, job}", 200

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8000, debug=True)

