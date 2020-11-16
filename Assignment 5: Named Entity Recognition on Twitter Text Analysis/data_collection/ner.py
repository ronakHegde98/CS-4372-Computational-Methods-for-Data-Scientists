from nltk.tag import StanfordNERTagger
from nltk.tokenize import word_tokenize
import nltk
import os

# nltk.download('punkt')

def classify(text):
  
    st = StanfordNERTagger('../stanford-ner-4.0.0/classifiers/english.all.3class.distsim.crf.ser.gz',
                        '../stanford-ner-4.0.0/stanford-ner-4.0.0.jar',
                        encoding='utf-8')

    tokenized_text = word_tokenize(text)
    classified_text = st.tag(tokenized_text)

    return (classified_text)


def named_entity_recognition(tweets):
    """named entity recognition for a batch of tweets"""

    st = StanfordNERTagger('../stanford-ner-4.0.0/classifiers/english.all.3class.distsim.crf.ser.gz',
                        '../stanford-ner-4.0.0/stanford-ner-4.0.0.jar',
                        encoding='utf-8')
    for tweet in tweets:
        tokenized_text = word_tokenize(tweet)
        classified_text = st.tag(tokenized_text)

results = classify('While in France, Christine Lagarde discussed short-term stimulus efforts in a recent interview with the Wall Street Journal.')



