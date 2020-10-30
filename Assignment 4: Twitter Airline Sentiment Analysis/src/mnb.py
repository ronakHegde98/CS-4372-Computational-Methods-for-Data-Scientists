from sklearn.model_selection import train_test_split
from TweetPreprocessor import TweetPreprocessor
from sklearn.naive_bayes import MultinomialNB
import pandas as pd
import numpy as np
import sys
import os



if __name__ == "__main__":
    if len(sys.argv) == 2:
        file_location = sys.argv[1].strip()
        if(os.path.exists(file_location) and file_location.endswith("Tweets.csv")):
            columns = ['airline_sentiment', 'airline', 'text']
            tweets_df = pd.read_csv(file_location, usecols = columns)          
            target_col = 'airline_sentiment'

            preprocessor = TweetPreprocessor(tweets_df)
            normalized_counts, df = preprocessor.fit(tweet_col = 'text', target_col = target_col)
            X_train, X_test, y_train, y_test = train_test_split(normalized_counts, df[target_col], test_size=0.1)


        else:
            print("Error Processing File; Please enter valid Tweets.csv file")
    else:
        print("Please enter the file path")