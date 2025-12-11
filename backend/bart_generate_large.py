# not successful, but included for language generation attempt 

import torch
from transformers import BartForConditionalGeneration, BartTokenizer
from backend.bert_similarity import extract_keywords, embed_sentence, load_bert

# load models
bart_model = BartForConditionalGeneration.from_pretrained("facebook/bart-large")
bart_tokenizer = BartTokenizer.from_pretrained("facebook/bart-large")
bert_tokenizer, bert_model = load_bert()

def semantic_similarity(text1, text2):
    embedding_1 = embed_sentence(text1, bert_tokenizer, bert_model)
    embedding_2 = embed_sentence(text2, bert_tokenizer, bert_model)

    similarity = torch.nn.functional.cosine_similarity(embedding_1, embedding_2).item()
    return similarity

def paraphrase_phrase(jd_phrase, resume_phrase):
    # input prompt attempt
    prompt = (
        f"Paraphrase this resume phrase to better match the job description phrase.\n"
        f"JD phrase: {jd_phrase}\n"
        f"Resume phrase: {resume_phrase}\n"
    )

    # pass prompt to tokenizer
    inputs = bart_tokenizer([prompt], return_tensors="pt", truncation=True, padding=True)
    
    # generate language
    generated_ids = bart_model.generate(
        inputs["input_ids"],
        num_beams=4,
        max_length=50,
        early_stopping=True,
        no_repeat_ngram_size=3
    )
    return bart_tokenizer.decode(generated_ids[0], skip_special_tokens=True)

def rewrite_resume_line(line, substitutions):
    for resume_phrase, suggestion in substitutions.items():
        if resume_phrase.lower() in line.lower():
            index = line.lower().find(resume_phrase.lower())
            line = line[:index] + suggestion + line[index + len(resume_phrase):]
    return line

def generate_bart_rewrites(jd_path, resume_path, similarity_threshold=0.75):
    jd_text = open(jd_path).read()
    resume_text = open(resume_path).read()
    resume_lines = [line.strip() for line in resume_text.split("\n") if line.strip()]

    jd_keywords = extract_keywords(jd_text, max_keywords=5)

    # for resume, jd pairs
    substitutions = {}
    for jd_phrase in jd_keywords:
        for line in resume_lines:
            score = semantic_similarity(jd_phrase, line)
            if score >= similarity_threshold:
                suggestion = paraphrase_phrase(jd_phrase, line)
                substitutions[jd_phrase] = suggestion

    rewritten_resume = [rewrite_resume_line(line, substitutions) for line in resume_lines]

    return rewritten_resume, substitutions
