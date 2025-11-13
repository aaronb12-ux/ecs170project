# Similarity Analysis

```
Note: Install NLTK and Sklearn libraries
pip install nltk
pip install scikit-learn
```
### Jaccard Similarity Analysis:
- It measures the similarity between two sets. It is calculated by dividing the size of the intersection of the two sets by the size of their union. It is often used to compare the similarity of two text documents or strings based on their shared words or tokens.
- Example:
```python
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk
import string

nltk.download('punkt_tab')
nltk.download("punkt")
nltk.download("stopwords")

def jaccard_similarity(text1, text2):
    stop = set(stopwords.words("english"))
    # tokenize and clean
    def clean(t):
        return {w.lower() for w in word_tokenize(t)
                if w.isalnum() and w.lower() not in stop}
    set1, set2 = clean(text1), clean(text2)
    return len(set1 & set2) / len(set1 | set2 or {1})

a = "The quick brown fox jumps over the lazy dog"
b = "A quick brown cat sleeps near a lazy dog"

print("Jaccard similarity:", round(jaccard_similarity(a, b), 3))

```

### WordNet-based Similarity Analysis:
- NLTK provides access to WordNet, a lexical database that groups English words into sets of synonyms called synsets and records relationships between them.
- Example:
```python
from nltk.corpus import wordnet as wn
import nltk
nltk.download("wordnet")
nltk.download("omw-1.4")

def wordnet_similarity(word1, word2):
    syns1 = wn.synsets(word1)
    syns2 = wn.synsets(word2)
    if not syns1 or not syns2:
        return 0
    # pick max path similarity among all sense pairs
    sims = [s1.path_similarity(s2) or 0 for s1 in syns1 for s2 in syns2]
    return max(sims)

print("Similarity(car, automobile):", wordnet_similarity("car", "automobile"))
print("Similarity(car, banana):", wordnet_similarity("car", "banana"))
```
  

### Hybrid Similarity Analysis:
- It is an integrated method that combines Jaccard similarity and WordNet-based semantic similarity.

### Cosine Similarity Analysis:
- The core idea of  cosine similarity is to represent text as numerical vectors and then calculate the cosine of the angle between these vectors. This can be achieved by using simpler methods like Bag-of-Words (BoW) or TF-IDF for vectorization.
- Example:
```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Sample texts
text1 = "The quick brown fox jumps over the lazy dog."
text2 = "A lazy cat sleeps under the warm sun."
text3 = "The quick brown fox runs fast."

# Preprocessing function
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text) # Remove punctuation
    tokens = text.split()
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    stemmer = PorterStemmer()
    tokens = [stemmer.stem(word) for word in tokens]
    return ' '.join(tokens)

# Preprocess the texts
processed_text1 = preprocess_text(text1)
processed_text2 = preprocess_text(text2)
processed_text3 = preprocess_text(text3)

# Create TF-IDF vectors
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform([processed_text1, processed_text2, processed_text3])

# Calculate cosine similarity
similarity_text1_text2 = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
similarity_text1_text3 = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[2:3])[0][0]

print(f"Cosine similarity between text1 and text2: {similarity_text1_text2}")
print(f"Cosine similarity between text1 and text3: {similarity_text1_text3}")
```

# Sentiment Analysis

VADER (Valence Aware Dictionary and sEntiment Reasoner)
- it uses a pre-defined dictionary where each word has a sentiment intensity score (positive, negative, neutral)

SentiWordNet
- it extends WordNet (a large lexical database) with sentiment scores for each synset (set of synonyms).

Pros:  
Easy to use - Sentiment analysis (VADER, for example) allows comparing documents based on normalized scores extracted from the emotional implications of words in the analyzed data. Two documents can be easily compared using their scores.  
Cons:  
Hard to apply - Sentiment analysis mainly focuses on emotions and opinions; typical applications of sentiment analysis include identifying opinions on subjects through user-generated reviews. This project mainly relies on matching documents based on semantic similarity, so sentiment analysis may be hard to use.  
              - Our application of sentiment analysis may include creating a list of words and their associated scores for the model to use in order to properly judge input text ([this is how VADER does its sentiment analysis](https://github.com/cjhutto/vaderSentiment#:~:text=Sentiment%20ratings%20from,are%20both%20%E2%80%931.5.)). However, since words only have a single score ranging between negative and positive, it would be difficult to represent a wide range of professions only using a single scalable score. Furthermore, there exists a type of sentiment analysis named aspect-based sentiment analysis, in which aspects of a subject are each analyzed separately rather than analyzing the subject as a whole. Still, it would be incredibly time consuming to create an array of professions and words that relate to each one, with each word given their own score to represent its connection to the profession.  

Fuzzy Search:

Lucene
- search library that allows text indexing and similarity matching. JAVA INITIALLY but has ports to other languages


# A simple way to install NLTK library:

```python
# for virtual environment
python -m venv venv

# activate virtual environment - virtual environment is not necessary
venv\Scripts\activate # Windows
source venv/bin/activate # macOS or Linux

# install nltk library inside virtual environment 
pip install nltk
if it doesn't work, then
pip3 install nltk

# when doen, deactivate
deactivate


# quick test for VADER
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk
nltk.download('vader_lexicon')

sia = SentimentIntensityAnalyzer()
print(sia.polarity_scores("This game is ridiculously good!!!"))

# expected output of the above code
{'neg': 0.262, 'neu': 0.327, 'pos': 0.412, 'compound': 0.3348}
```
