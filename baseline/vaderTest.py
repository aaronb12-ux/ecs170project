from nltk.sentiment import SentimentIntensityAnalyzer
import nltk
nltk.download('vader_lexicon')

sia = SentimentIntensityAnalyzer()

# 1) Add/override domain words and weights
custom_lexicon = {
    "python": 3.0,
    "django": 3.0,
    "flask": 3.0,
    "postgresql": 2.0,
    "mysql": 2.0,
    "docker": 1.0,
    "kubernetes": 1.0,
    "rest api": 1.0,
    "aws": 2.0,
    "azure": 2.0,
    "git": 1.0
}
sia.lexicon.update(custom_lexicon)

# 2) Score text
texts = [
    '''
    - Built REST APIs using Python and Django
    - Worked with PostgreSQL and Redis for data storage
    - Implemented CI/CD pipelines with Jenkins and Docker
    - Collaborated with frontend team on React integration
    '''
]
for t in texts:
    print(t, "->", sia.polarity_scores(t)["compound"])