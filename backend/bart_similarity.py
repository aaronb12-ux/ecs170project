
import torch
from transformers import BartTokenizer, BartForConditionalGeneration
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Load the BART model and tokenizer
tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')
model = BartForConditionalGeneration.from_pretrained('facebook/bart-large')

def preprocess_text(text):
    """
    Normalize the text: convert to lowercase.
    """
    return text.lower()

def compute_average_embeddings(sentences):
    """
    Compute the average embeddings for a list of sentences.
    """
    embeddings = []
    for sentence in sentences:
        inputs = tokenizer(sentence, return_tensors='pt', padding=True, truncation=True)

        with torch.no_grad():
            outputs = model(**inputs)

        # Use the encoder's last hidden state
        last_hidden_state = outputs.encoder_last_hidden_state
        sentence_embedding = last_hidden_state.mean(dim=1).numpy()  # Average the embeddings
        embeddings.append(sentence_embedding)

    # Stack and compute the overall average embedding
    return np.mean(np.vstack(embeddings), axis=0)

def compute_similarity(resume_text: str, job_description_text: str, sentence_threshold: float = 0.9) -> dict:
    """
    Compute BART-based similarity between resume and job description.
    Returns overall similarity and sentence-level overlaps above the threshold.
    """
    resume_text = preprocess_text(resume_text)
    job_description_text = preprocess_text(job_description_text)

    # Split text into sentences
    resume_sentences = [s.strip() for s in resume_text.split('.') if s.strip()]
    job_sentences = [s.strip() for s in job_description_text.split('.') if s.strip()]

    # Compute embeddings
    resume_embedding = compute_average_embeddings(resume_sentences)
    job_embedding = compute_average_embeddings(job_sentences)

    # Overall cosine similarity
    overall_similarity = cosine_similarity(resume_embedding.reshape(1, -1), job_embedding.reshape(1, -1))[0][0]

    # Sentence-level similarities
    sentence_matches = []
    for r_sentence in resume_sentences:
        r_vec = compute_average_embeddings([r_sentence])
        for j_sentence in job_sentences:
            j_vec = compute_average_embeddings([j_sentence])
            sim = cosine_similarity(r_vec.reshape(1, -1), j_vec.reshape(1, -1))[0][0]
            if sim >= sentence_threshold:
                sentence_matches.append({
                    "resume_sentence": r_sentence,
                    "job_sentence": j_sentence,
                    "similarity": sim
                })

    return {
        "overall_similarity": overall_similarity,
        "sentence_matches": sentence_matches
    }
