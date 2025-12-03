import json
import os
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

def clean_text(text):
    text = text.lower()
    # Remove special characters and digits
    text = re.sub(r'[^a-z\s\+\#\.\/]', '', text)
    return text

def analyze_resume(resume_text, jd_text):
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
    match_percentage = round(match_percentage * 100, 2)
    
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

    # Generate feedback
    feedback_text = ""
    if match_percentage >= 80:
        feedback_text = "Excellent match!"
    elif match_percentage >= 50:
        feedback_text = "Good match."
    else:
        feedback_text = "Low match."

    # Construct result disctionary for JSON file
    result = {
        "match_percentage": f"{match_percentage}%",
        "match_score_raw": match_percentage,
        "feedback": feedback_text,
        "missing_keywords": missing_keywords,
        "top_keywords_in_jd": top_jd_keywords
    }
    
    return result


if __name__ == "__main__":

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

    # Run analysis
    analysis_result = analyze_resume(user_resume, job_description)

    # Print them for verification
    print("Analysis Complete.")
    print(f"Score: {analysis_result['match_percentage']}")
    print(f"Missing Keywords: {analysis_result['missing_keywords']}")

    # Get current directory
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Join that directory
    output_path = os.path.join(script_dir, "resume_feedback.json")

    # Output JSON file
    with open(output_path, "w") as json_file:
        json.dump(analysis_result, json_file, indent=4)
    
    print(f"Data saved to {output_path}")