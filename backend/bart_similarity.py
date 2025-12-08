import re
import torch
from transformers import BartTokenizer, BartModel
from sklearn.metrics.pairwise import cosine_similarity

tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')
model = BartModel.from_pretrained('facebook/bart-large')  

def embed_sentence(sentence, tokenizer, model):
    inputs = tokenizer(sentence, return_tensors="pt", truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
    emb = outputs.last_hidden_state.mean(dim=1)
    return emb

def embed_lines(lines, tokenizer, model):
    embeddings = []
    for line in lines:
        embeddings.append(embed_sentence(line, tokenizer, model))
    return embeddings

def compute_file_similarity(jd_matrix, resume_matrix):
    return cosine_similarity(
        jd_matrix.mean(axis=0, keepdims=True),
        resume_matrix.mean(axis=0, keepdims=True)
    )[0][0]

def get_similar_pairs(lines_jd, emb_jd, lines_resume, emb_resume, th=0.75):
    pairs = []
    for i, e_jd in enumerate(emb_jd):
        for j, e_res in enumerate(emb_resume):
            score = cosine_similarity(
                e_jd.detach().numpy(),
                e_res.detach().numpy()
            )[0][0]
            if score >= th:
                pairs.append((i, j, score))
    return pairs

def compute_similarity(resume_text: str, job_description_text: str, sentence_threshold: float = 0.80) -> dict:
    lines_jd = [line.strip() for line in job_description_text.splitlines() if line.strip()]
    lines_resume = [line.strip() for line in resume_text.splitlines() if line.strip()]

    emb_resume = embed_lines(lines_resume, tokenizer, model)
    emb_jd = embed_lines(lines_jd, tokenizer, model)

    jd_matrix = torch.cat(emb_jd, dim=0).detach().numpy()
    resume_matrix = torch.cat(emb_resume, dim=0).detach().numpy()

    file_score = compute_file_similarity(jd_matrix, resume_matrix)
    pairs = get_similar_pairs(lines_jd, emb_jd, lines_resume, emb_resume, sentence_threshold)

    return {
        "file_similarity": file_score,
        "lines_jd": lines_jd,
        "lines_resume": lines_resume,
        "similar_pairs": pairs
    }

def format_bart_json_sorted(analysis):
    jd_to_best_match = {}
    for jd_idx, res_idx, score in analysis["similar_pairs"]:
        if jd_idx not in jd_to_best_match or score > jd_to_best_match[jd_idx][1]:
            jd_to_best_match[jd_idx] = (analysis["lines_resume"][res_idx], score)

    strengths_tuples = [
        (analysis["lines_jd"][jd_idx], res_line, score)
        for jd_idx, (res_line, score) in jd_to_best_match.items()
    ]
    strengths_tuples.sort(key=lambda x: x[2], reverse=True)

    strengths = [f"{jd} | {res} | similarity: {score:.4f}" for jd, res, score in strengths_tuples]

    matched_jd_indices = set(jd_to_best_match.keys())
    weaknesses = [
        line for i, line in enumerate(analysis["lines_jd"]) if i not in matched_jd_indices
    ]

    return {
        "similarity_score": float(analysis["file_similarity"]),
        "strengths": strengths,
        "weaknesses": weaknesses
    }
