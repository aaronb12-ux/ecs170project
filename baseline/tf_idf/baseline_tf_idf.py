import json
import os
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import docx2txt

# Default resume and job description definitions in case not input
job_description = """
    Software Engineer Position

    We're looking for a backend engineer to join our team.

    Requirements:
    - 2+ years experience with Python
    - Strong knowledge of Django or Flask
    - Experience with PostgreSQL or MySQL
    - Familiarity with Docker and Kubernetes
    - Understanding of REST API design
    - Experience with AWS or Azure cloud platforms
    - Git version control

    Nice to have:
    - React or Vue.js experience
    - Redis caching
    - CI/CD pipeline setup 
    - C++
    """

user_resume = """
    Sarah Chen
    Software Engineer
    sarah.chen@email.com

    EXPERIENCE
    Backend Engineer at DataFlow Inc (2022-2024)
    - Built REST APIs using Python and Django
    - Worked with PostgreSQL and Redis for data storage
    - Implemented  pipelines with Jenkins and Docker
    - Collaborated with frontend team on React integration

    SKILLS
    Python, JavaScript, Django, Flask, PostgreSQL, Docker, Git, AWS
    """

def clean_text(text):
    text = text.lower()
    # Remove special characters and digits
    text = re.sub(r'[^a-z\s\+\#\.\/]', '', text)
    return text

def print_results(analysis_result):
    # Print them for verification
    print("Analysis Complete.")
    print(f"Strengths: {analysis_result['strengths']}")
    print(f"Weaknesses: {analysis_result['weaknesses']}")

    # Get current directory
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Join that directory
    output_path = os.path.join(script_dir, "baseline_results.json")

    # Output JSON file
    with open(output_path, "w") as json_file:
        json.dump(analysis_result, json_file, indent=4)
    
    print(f"Data saved to {output_path}")

# The values are initialized to defaults in case none are given from
# an external function call
def run_baseline_tfidf(resume_text=user_resume, jd_text=job_description, print=False):
    # Clean texts
    clean_resume = clean_text(resume_text)
    clean_jd = clean_text(jd_text)
    
    # Get TF-IDF vectors for both text
    vectorizer = TfidfVectorizer(stop_words='english', token_pattern=r'\S+')
    
    # Get TF-IDF matrix for cosine similarity
    tfidf_matrix = vectorizer.fit_transform([clean_resume, clean_jd])
    
    # Compute Cosine Similarity
    # tfidf_matrix[0] -> resume
    # tfidf_matrix[1] -> job description
    match_percentage = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
    match_percentage = round(match_percentage, 2)
    
    # Identify missing keywords
    feature_names = vectorizer.get_feature_names_out()
    
    # Get the vector for job description
    jd_vector = tfidf_matrix[1]
    
    # Create a dataframe to sort keywords by importance
    df = pd.DataFrame(jd_vector.T.todense(), index=feature_names, columns=["tfidf"])
    
    # Sort by score descending
    sorted_keywords = df.sort_values(by=["tfidf"], ascending=False)
    
    # Get only top 20 keywords
    top_jd_keywords = sorted_keywords[sorted_keywords['tfidf'] > 0].head(20).index.tolist()
    
    # Check which top 20 keywords are missing from the resume
    missing_keywords = [word for word in top_jd_keywords if word not in clean_resume]

    # Check which top 20 keywords are in both the jd and resume
    matching_keywords = [word for word in top_jd_keywords if word in clean_resume]

    # Construct result disctionary for JSON file
    result = {
        "similarityScore": f"{match_percentage}",
        "strengths": matching_keywords,
        "weaknesses": missing_keywords
    }

    if(print):
        print_results(result)
    
    return result


if __name__ == "__main__":

    # For testing
    try:
        # Check for a document named "resume.docx"
        resume = docx2txt.process("resume.docx")
    except:
        # If not present, use the default template resume
        resume = user_resume

    try:
        # Check for "job_description"
        job_desc = docx2txt.process("job_description.docx")
    except:
        # If not in file structure, use default job description global variable
        job_desc = job_description

    # Run analysis
    analysis_result = run_baseline_tfidf(resume, job_desc, True)