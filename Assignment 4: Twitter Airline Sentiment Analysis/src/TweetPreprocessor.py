from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import preprocessor as p
import pandas as pd
import numpy as np



class TweetPreprocessor:
    def __init__(self, df):
        self.df = df
    
    def fit(self, tweet_col, target_col):
        np.random.shuffle(self.df.values)
        self.drop_nulls()
        self.lower_case(column_name = tweet_col)
        self.preprocess_tweets(tweet_col)
        self.label_encode(target_col)
        
        count_vect = CountVectorizer()
        counts = count_vect.fit_transform(self.df[tweet_col])
        transformer = TfidfTransformer().fit(counts)
        normalized_counts = transformer.transform(counts)  
        
        return normalized_counts, self.df,  
    
    def label_encode(self, target_col):
        le = LabelEncoder()
        self.df[target_col] = le.fit_transform(self.df[target_col])
    
    def preprocess_tweets(self, tweet_col):
        self.df[tweet_col] = self.df[tweet_col].apply(p.clean)
        
    def token_counts(self, tweet_col):
        count_vect = CountVectorizer()
        counts = count_vect.fit_transform(self.df[tweet_col])
        return counts
    
    def normalize_counts(self, counts):
        transformer = TfidfTransformer().fit(counts)
        return transformer.transform(counts)
    
    def drop_nulls(self):
        if(self.df.isna().sum().sum()>0):
            self.df.dropna(inplace=True)
        
    def lower_case(self, column_name):
        self.df[column_name] = self.df[column_name].str.lower()