# ref: https://huggingface.co/docs/transformers/en/model_doc/bert, section: BertForPreTraining
# ref (embeddings): https://www.geeksforgeeks.org/nlp/how-to-generate-word-embedding-using-bert/
# ref: https://www.geeksforgeeks.org/python/how-to-read-from-a-file-in-python/ 
# ref (tokenize text into sentences): https://www.geeksforgeeks.org/nlp/tokenize-text-using-nltk-python/

import re
import torch
from transformers import BertTokenizer, BertModel
from sklearn.metrics.pairwise import cosine_similarity

# model import
def load_bert():
    tokenizer = BertTokenizer.from_pretrained("google-bert/bert-base-uncased")
    model = BertModel.from_pretrained("google-bert/bert-base-uncased")
    return tokenizer, model

# process txt files into sentences
def load_file(path):
    with open(path, "r") as f:
        return f.read()

# clean text    
def clean_lines(text, min_length=5):
    lines = text.splitlines()
    return [ln.strip() for ln in lines if len(ln.strip()) > min_length]
   
# embeddings 

def embed_sentence(sentence, tokenizer, model):
    inputs = tokenizer(sentence, return_tensors="pt", truncation=True)
    outputs = model(**inputs)
    emb = outputs.last_hidden_state.mean(dim=1)
    return emb

def embed_lines(lines, tokenizer, model):
    embeddings = []
    for sent in lines:
        embeddings.append(embed_sentence(sent, tokenizer, model))
    return embeddings

# cosine similarity 
def compute_file_similarity(jd_matrix, resume_matrix):
    return cosine_similarity(
        jd_matrix.mean(axis=0, keepdims=True),
        resume_matrix.mean(axis=0, keepdims=True)
    )[0][0]


def get_similar_pairs(lines_jd, embeddings_jd, lines_resume, embeddings_resume, th=0.75):
    pairs = []
    for i, e_jd in enumerate(embeddings_jd):
        for j, e_res in enumerate(embeddings_resume):
            score = cosine_similarity(
                e_jd.detach().numpy(),
                e_res.detach().numpy()
            )[0][0]
            if score >= th:
                pairs.append((i, j, score))
    return pairs

# get keywords
def extract_keywords(jd_sentence, max_keywords=3):
    words = re.findall(r"[A-Za-z]+", jd_sentence.lower())
    stop = {
        "and","the","for","with","that","from","this","will","your",
        "into","using","ability","skills","experience","work"
    }
    filtered = [w for w in words if len(w) > 4 and w not in stop]
    return filtered[:max_keywords]

# return similarity scores/pairs
def analyze_files(jd_text, resume_text, th=0.75):

    # Clean lines
    lines_jd = clean_lines(jd_text)
    lines_resume = clean_lines(resume_text)

    # Load embedding model
    tokenizer, model = load_bert()

    # Generate embeddings
    emb_jd = embed_lines(lines_jd, tokenizer, model)
    emb_resume = embed_lines(lines_resume, tokenizer, model)

    # Build matrices
    jd_matrix = torch.cat(emb_jd, dim=0).detach().numpy()
    resume_matrix = torch.cat(emb_resume, dim=0).detach().numpy()

    # File similarity
    file_score = compute_file_similarity(jd_matrix, resume_matrix)

    # Sentence-level matches
    pairs = get_similar_pairs(lines_jd, emb_jd, lines_resume, emb_resume, th)

    return {
        "file_similarity": file_score,
        "lines_jd": lines_jd,
        "lines_resume": lines_resume,
        "similar_pairs": pairs
    }

