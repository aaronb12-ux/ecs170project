from baseline.text_overlap import overlap_analysis
from backend.bert_similarity import analyze_files, format_similarity_json_sorted
from backend.word2vec_similarity import run_w2v, format_w2v_json_sorted
from backend.bart_similarity import compute_similarity, format_bart_json_sorted
import json

def run_bert_similarity(resume_path, jd_path):
    with open(resume_path, "r") as f:
        resume = f.read()
    with open(jd_path, "r") as f:
        job = f.read()
    analysis = analyze_files(job, resume) 
    return format_similarity_json_sorted(analysis)


def run_w2v_similarity(resume_path, jd_path):
    with open(resume_path, "r") as f:
        resume = f.read()
    with open(jd_path, "r") as f:
        job = f.read()
    analysis = run_w2v(resume, job)
    return format_w2v_json_sorted(analysis)


def run_bart_similarity(resume_path, jd_path):
    with open(resume_path, "r") as f:
        resume = f.read()
    with open(jd_path, "r") as f:
        job = f.read()
    analysis = compute_similarity(resume, job)
    return format_bart_json_sorted(analysis)


def main():
    resume_path = "sample_text/resume1.txt"
    jd_path = "sample_text/jd1.txt"

    json_output_bert = run_bert_similarity(resume_path, jd_path)
    json_output_w2v = run_w2v_similarity(resume_path, jd_path)
    json_output_bart = run_bart_similarity(resume_path, jd_path)

    print("=== BERT Similarity ===")
    print(json.dumps(json_output_bert, indent=4))

    print("\n=== Word2Vec Similarity ===")
    print(json.dumps(json_output_w2v, indent=4))

    print("\n=== BART Similarity ===")
    print(json.dumps(json_output_bart, indent=4))


if __name__ == "__main__":
    main()
