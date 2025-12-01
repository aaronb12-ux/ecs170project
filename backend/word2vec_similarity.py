# ref: https://www.geeksforgeeks.org/python/python-word-embedding-using-word2vec/
import gensim.downloader as api
from gensim.utils import simple_preprocess
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# load pre-trained word2vec model - takes a long time to download on first time
# smaller model (100-dim): "glove-wiki-gigaword-100"
model = api.load("word2vec-google-news-300")  # 300-dim embeddings

# read files
with open("jobdescription.txt", "r") as f:
    content_jd = f.read()

with open("resume.txt", "r") as f:
    content_resume = f.read()

# split into lines / sentences
lines_jd_raw = content_jd.splitlines()
lines_resume_raw = content_resume.splitlines()

# remove short or empty lines from job description
lines_jd = []
for line in lines_jd_raw:
    stripped_line = line.strip()  # remove leading/trailing whitespace
    #if len(stripped_line) > 5:   # only keep lines longer than 5 characters
    lines_jd.append(stripped_line)

# remove short or empty lines from resume
lines_resume = []
for line in lines_resume_raw:
    stripped_line = line.strip()  # remove leading/trailing whitespace
    #if len(stripped_line) > 5:   # only keep lines longer than 5 characters
    lines_resume.append(stripped_line)

embeddings_jd = []
for sentence in lines_jd:
    words = simple_preprocess(sentence)  
    word_vecs = []
    for word in words:
        if word in model:
            word_vecs.append(model[word])
    if len(word_vecs) == 0:  # if no words in vocab, return zeros
        sentence_embedding = np.zeros(model.vector_size)
    else:
        sentence_embedding = np.mean(word_vecs, axis=0)
    embeddings_jd.append(sentence_embedding)

# generate embeddings for resume
embeddings_resume = []
for sentence in lines_resume:
    words = simple_preprocess(sentence)  
    word_vecs = []
    for word in words:
        if word in model:
            word_vecs.append(model[word])
    if len(word_vecs) == 0:  # if no words in vocab, return zeros
        sentence_embedding = np.zeros(model.vector_size)
    else:
        sentence_embedding = np.mean(word_vecs, axis=0)
    embeddings_resume.append(sentence_embedding)

# average embeddings for overall similarity
jd_matrix = np.vstack(embeddings_jd)
resume_matrix = np.vstack(embeddings_resume)

overall_score = cosine_similarity(jd_matrix.mean(axis=0, keepdims=True),
                                  resume_matrix.mean(axis=0, keepdims=True))[0][0]

print(f"Overall file similarity score: {overall_score:.4f}")

# generate similarity scores for individual sentences
for i, emb_jd in enumerate(embeddings_jd):
    for j, emb_resume in enumerate(embeddings_resume):
        score = cosine_similarity(emb_jd.reshape(1, -1), emb_resume.reshape(1, -1))[0][0]
        if score > 0.55:
            print(f"\nSimilarity score: {score:.4f}")
            print(f"Job description sentence: {lines_jd[i]}")
            print(f"Resume sentence: {lines_resume[j]}")

