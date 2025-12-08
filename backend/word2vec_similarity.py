# ref: https://www.geeksforgeeks.org/python/python-word-embedding-using-word2vec/
import gensim.downloader as api
from gensim.utils import simple_preprocess
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# load pre-trained word2vec model - takes a long time to download on first time
# smaller model (100-dim): "glove-wiki-gigaword-100"
model = api.load("word2vec-google-news-300") # 300-dim embeddings

def read_file_lines(file_path):
    with open(file_path, "r") as f:
        lines = [line.strip() for line in f.readlines()]
    return lines


def sentence_embeddings(lines, model):
    embeddings = []
    for sentence in lines:
        words = simple_preprocess(sentence)
        word_vecs = [model[word] for word in words if word in model]
        if len(word_vecs) == 0:
            sentence_embedding = np.zeros(model.vector_size)
        else:
            sentence_embedding = np.mean(word_vecs, axis=0)
        embeddings.append(sentence_embedding)
    return embeddings

# compute cosine similarity for avg embeddings of two sets of sentences
def overall_file_similarity(embeddings1, embeddings2):
    matrix1 = np.vstack(embeddings1)
    matrix2 = np.vstack(embeddings2)
    score = cosine_similarity(matrix1.mean(axis=0, keepdims=True),
                              matrix2.mean(axis=0, keepdims=True))[0][0]
    return score

def sentence_level_similarity(lines_jd, embeddings_jd, lines_resume, embeddings_resume, th=0.55):
    results = []
    for i, emb_jd in enumerate(embeddings_jd):
        for j, emb_resume in enumerate(embeddings_resume):
            score = cosine_similarity(emb_jd.reshape(1, -1), emb_resume.reshape(1, -1))[0][0]
            if score > th:
                results.append({
                    "jd_sentence": lines_jd[i],
                    "resume_sentence": lines_resume[j],
                    "similarity": score
                })
    return results

def run_w2v(resume_text, jd_text, th=0.55):

    lines_jd = [line.strip() for line in jd_text.splitlines() if line.strip()]
    lines_resume = [line.strip() for line in resume_text.splitlines() if line.strip()]

    embeddings_jd = sentence_embeddings(lines_jd, model)
    embeddings_resume = sentence_embeddings(lines_resume, model)

    overall_score = overall_file_similarity(embeddings_jd, embeddings_resume)
    sentence_matches = sentence_level_similarity(lines_jd, embeddings_jd, lines_resume, embeddings_resume, th)

    return {
        "overall_score": overall_score,
        "sentence_matches": sentence_matches,
        "all_jd_lines": lines_jd,      
        "all_resume_lines": lines_resume
    }

def format_w2v_json_sorted(analysis):
    jd_to_best_match = {}
    for match in analysis["sentence_matches"]:
        jd_line = match["jd_sentence"]
        res_line = match["resume_sentence"]
        score = float(match["similarity"])
        if jd_line not in jd_to_best_match or score > jd_to_best_match[jd_line][1]:
            jd_to_best_match[jd_line] = (res_line, score)
    
    strengths_tuples = [(jd, res, score) for jd, (res, score) in jd_to_best_match.items()]
    strengths_tuples.sort(key=lambda x: x[2], reverse=True)
    strengths = [f"{jd} | {res} | similarity: {score:.4f}" for jd, res, score in strengths_tuples]

    all_jd_lines = set([match["jd_sentence"] for match in analysis["sentence_matches"]])
    weaknesses = [jd for jd in analysis.get("all_jd_lines", []) if jd not in all_jd_lines]

    return {
        "similarity_score": float(analysis["overall_score"]),
        "strengths": strengths,
        "weaknesses": weaknesses
    }

