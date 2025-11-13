# Similarity Analysis

Jaccard Similarity Analysis:
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

WordNet-based Similarity Analysis:
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
  

Hybrid Similarity Analysis:
- It is an integrated method that combines Jaccard similarity and WordNet-based semantic similarity.

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
