# ref: https://huggingface.co/docs/transformers/en/model_doc/bert, section: BertForPreTraining
# ref (embeddings): https://www.geeksforgeeks.org/nlp/how-to-generate-word-embedding-using-bert/
# ref: https://www.geeksforgeeks.org/python/how-to-read-from-a-file-in-python/ 
# ref (tokenize text into sentences): https://www.geeksforgeeks.org/nlp/tokenize-text-using-nltk-python/

import torch
from transformers import BertTokenizer, BertModel
from sklearn.metrics.pairwise import cosine_similarity

# model import
tokenizer = BertTokenizer.from_pretrained("google-bert/bert-base-uncased")
model = BertModel.from_pretrained("google-bert/bert-base-uncased") 

# process txt files into sentences
file_jd = open("jobdescription.txt", "r")
content_jd = file_jd.read()
file_jd.close()

file_resume = open("resume.txt", "r")
content_resume = file_resume.read()
file_resume.close()

lines_jd_raw = content_jd.splitlines()
lines_resume_raw = content_resume.splitlines()

# remove short or empty lines
lines_jd = []
for line in lines_jd_raw:
    stripped_line = line.strip()
    if len(stripped_line) > 5:
        lines_jd.append(stripped_line)

lines_resume = []
for line in lines_resume_raw:
    stripped_line = line.strip()
    if len(stripped_line) > 5:
        lines_resume.append(stripped_line)

# generate embeddings 
embeddings_jd = []
for sentence in lines_jd:
    inputs = tokenizer(sentence, return_tensors="pt", truncation=True)
    outputs = model(**inputs)
    word_embeddings = outputs.last_hidden_state
    sentence_embedding = word_embeddings.mean(dim=1)
    embeddings_jd.append(sentence_embedding)

embeddings_resume = []
for sentence in lines_resume:
    inputs = tokenizer(sentence, return_tensors="pt", truncation=True)
    outputs = model(**inputs)
    word_embeddings = outputs.last_hidden_state
    sentence_embedding = word_embeddings.mean(dim=1)
    embeddings_resume.append(sentence_embedding)


# avg all sentence embeddings 
jd_matrix = torch.cat(embeddings_jd, dim=0).detach().numpy()  # shape: (num_jd_sentences, hidden_size)
resume_matrix = torch.cat(embeddings_resume, dim=0).detach().numpy()  # shape: (num_resume_sentences, hidden_size)

overall_score = cosine_similarity(jd_matrix.mean(axis=0, keepdims=True),
                                  resume_matrix.mean(axis=0, keepdims=True))[0][0]

print(f"Overall file similarity score: {overall_score:.4f}")

# generate similarity scores using cosine similarity 
for i, embedding_jd in enumerate(embeddings_jd):
    for j, embedding_resume in enumerate(embeddings_resume):
        # pass vectors as inputs
        score = cosine_similarity(embedding_jd.detach().numpy(), embedding_resume.detach().numpy())[0][0]
        
        if score > 0.75: # around 0.5, get matches between job titles and description lines - not useful 
            print(f"\nSimilarity score: {score:.4f}")
            print(f"Job description sentence: {lines_jd[i]}")
            print(f"Resume sentence: {lines_resume[j]}")
