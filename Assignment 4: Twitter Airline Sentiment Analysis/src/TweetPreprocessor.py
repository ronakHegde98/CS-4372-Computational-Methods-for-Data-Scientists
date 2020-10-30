"""
   TweetPreprocessor Class preprocesses a DataFrame with a tweet
   column and target column and gets it ready for sentiment
   analysis using Multinomial Naive Bayes

    Written by: Ronak Hegde and Vignesh Vasan starting 10/28/20
"""

from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer
from sklearn.preprocessing import LabelEncoder
import preprocessor as p
import pandas as pd
import numpy as np

class TweetPreprocessor:
    """ Preprocess Tweets Dataframe prior to Multinomial Naive Bayes """

    def __init__(self, df):
        self.df = df
    
    def fit(self, tweet_col, target_col):
        """
        Preprocessing pipeline that returns the 
        vectorized data and dataframe

        Keyword arguments:
        tweet_col -- DataFrame column with tweet text in it
        target_col -- DataFrame target column for algorithm
        """
        np.random.shuffle(self.df.values)
        self.drop_nulls()
        self.lower_case(column_name = tweet_col)
        self.preprocess_tweets(tweet_col)
        self.label_encode(target_col)
        
        counts = self.token_counts(tweet_col)
        normalized_counts = self.normalize_counts(counts)
        return normalized_counts, self.df

    def label_encode(self, target_col):
        """convert categorical target column to numbers"""
        le = LabelEncoder()
        self.df[target_col] = le.fit_transform(self.df[target_col])
    
    def preprocess_tweets(self, tweet_col):
        """Call Twitter-preprocessor library to remove emojis, mentions, urls"""
        self.df[tweet_col] = self.df[tweet_col].apply(p.clean)
        
    def token_counts(self, tweet_col):
        """return matrix of token counts from tweet column"""
        count_vect = CountVectorizer()
        return count_vect.fit_transform(self.df[tweet_col])
    
    def normalize_counts(self, counts):
        transformer = TfidfTransformer().fit(counts)
        return transformer.transform(counts)
    
    def drop_nulls(self):
        """Drop rows with null values if they exist"""
        if(self.df.isna().sum().sum()>0):
            self.df.dropna(inplace=True)
        
    def lower_case(self, column_name):
        """Lowercase the text of an entire column """
        self.df[column_name] = self.df[column_name].str.lower()