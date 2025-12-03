from pprint import pprint
from baseline.text_overlap import overlap_analysis
from backend.bert_similarity import analyze_files
from backend.word2vec_similarity import run_w2v as w2v_run
from backend.bart_similarity import compute_similarity
#from backend.bart_generate import generate_replacement_lines, print_rewrites
from backend.flan_generate import generate_replacement_lines, print_rewrites


def run_baseline_overlap(resume_path, jd_path):
    with open(resume_path, "r") as f:
        resume = f.read()
    with open(jd_path, "r") as f:
        job = f.read()
    result = overlap_analysis(resume, job)
    if 'baseline_similarity' in result:
        result['baseline_similarity'] = float(result['baseline_similarity'])
    return result

def run_bart_similarity(resume_path, jd_path):
    with open(resume_path, "r") as f:
        resume = f.read()
    with open(jd_path, "r") as f:
        job = f.read()
    return compute_similarity(resume, job)

# def run_bart_rewrite(resume_path, jd_path):
#     with open(resume_path, "r") as f:
#         resume = f.read()
#     with open(jd_path, "r") as f:
#         job = f.read()
#     result = generate_replacement_lines(job, resume)
#     for item in result:
#         item['similarity'] = float(item.get('similarity', 0))
#     return result

def run_bert(resume_path, jd_path):
    with open(resume_path, "r") as f:
        resume = f.read()
    with open(jd_path, "r") as f:
        job = f.read()
    result = analyze_files(job, resume)# jd_text first
    result['file_similarity'] = float(result['file_similarity'])
    for i, pair in enumerate(result['similar_pairs']):
        # (jd_index, resume_index, similarity)
        jd_idx, resume_idx, sim = pair
        result['similar_pairs'][i] = (jd_idx, resume_idx, float(sim))
    return result

def run_w2v(resume_path, jd_path):
    with open(resume_path, "r") as f:
        resume = f.read()
    with open(jd_path, "r") as f:
        job = f.read()
    result = w2v_run(resume, job)
    result['overall_score'] = float(result['overall_score'])
    for match in result.get('sentence_matches', []):
        match['similarity'] = float(match['similarity'])
    return result
    

def main():
    resume_path = "sample_text/resume1.txt"
    jd_path = "sample_text/jd1.txt"

    # print("=== BASELINE OVERLAP ===")
    # pprint(run_baseline_overlap(resume_path, jd_path), width=120)

    # print("\n=== BERT SIMILARITY ===")
    # pprint(run_bert(resume_path, jd_path), width=120)

    # print("\n=== BART SIMILARITY ===")
    # pprint(run_bart_similarity(resume_path, jd_path), width=120)

    # print("\n=== WORD2VEC SIMILARITY ===")
    # pprint(run_w2v(resume_path, jd_path), width=120)

    # print("\n=== BART REWRITE ===")
    # rewrites = run_bart_rewrite(resume_path, jd_path)
    # print_rewrites(rewrites)
    
    print("\n=== FLAN REWRITE ===")
    rewritten_lines = generate_replacement_lines(jd_path, resume_path, th=0.75)
    print_rewrites(rewritten_lines)

if __name__ == "__main__":
    main()


