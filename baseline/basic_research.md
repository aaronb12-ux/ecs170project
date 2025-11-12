Sentiment Analysis:

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
