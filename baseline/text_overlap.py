from baseline.skipwords import skipWords

def read_file(file_name):
    try:
        with open(file_name, 'r') as f:
            return f.read()
    except FileNotFoundError as e:
        return {"error": str(e)}
    
def preprocess_words(text: str):
    # lowercase and strip punctuation, return list of tokens
    tokens = text.split()
    cleaned = [t.lower().strip(".,:\n()[]{}!?;\"'") for t in tokens]
    return cleaned

def get_matching_words(resume_tokens, jd_tokens):
    matching = []

    jd_set = set(jd_tokens)  # faster lookup

    for item in resume_tokens:
        if item in jd_set and item not in skipWords:
            matching.append(item)
    
    return matching

def baseline_similarity_score(resume_tokens, jd_tokens):
    # similarity = (# of matching words) / (min(len(resume_tokens), len(jd_tokens)))
    matches = get_matching_words(resume_tokens, jd_tokens)
    denom = min(len(resume_tokens), len(jd_tokens))
    return len(matches) / denom if denom > 0 else 0.0


def overlap_analysis(resume_text: str, jd_text: str):
    # return keyword overlap + baseline score
    resume_tokens = preprocess_words(resume_text)
    jd_tokens = preprocess_words(jd_text)

    matching = get_matching_words(resume_tokens, jd_tokens)
    score = baseline_similarity_score(resume_tokens, jd_tokens)

    return {
        "matching_words": matching,
        "baseline_similarity": score
    }
