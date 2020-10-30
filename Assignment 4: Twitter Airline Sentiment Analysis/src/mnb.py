"""
   MultiNomial Naive Bayes Classifier using Sklearn to determine
   airline sentiment on Tweet dataset  
    Written by: Ronak Hegde and Vignesh Vasan starting 10/28/20
"""


from sklearn.model_selection import train_test_split
from TweetPreprocessor import TweetPreprocessor
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
import pandas as pd
import numpy as np
import sys
import os


def preprocess_tweets_df(file_location):
    columns = ['airline_sentiment', 'airline', 'text']
    target_col = columns[0]
    tweets_df = pd.read_csv(file_location, usecols = columns) 
    
    preprocessor = TweetPreprocessor(tweets_df)
    normalized_counts, df = preprocessor.fit(tweet_col = 'text', target_col = target_col)
    return normalized_counts, df

def mnb_optimizer(num_searches):
    """ Multinomial Naive Bayes optimizer will run (num_searches) amount of times to find 
    optimal values for hyperparameters for sklearn's MNB algorithm"""

    alpha_values = list(np.round(np.random.random_sample(num_searches),2))
    fit_prior_value = False
    accuracies = []

    for alpha_val in alpha_values:
        model = MultinomialNB(alpha=alpha_val, fit_prior = fit_prior_value).fit(X_train, y_train)
        predicted = model.predict(X_test)
        accuracy = np.round(np.mean(predicted == y_test),2)
        accuracies.append(accuracy)
    
    #all results go into dictionary fed into pandas DataFrame
    results = {'alpha':alpha_values, 'fit_prior': len(accuracies)*[False], 'accuracy': accuracies}
    results_df = pd.DataFrame(data = results)

    #find rows associated with max accuracy 
    max_accuracy = results_df['accuracy'].max()
    max_row = results_df[results_df['accuracy'] == max_accuracy]
    
    #take the alpha value from the first of these rows
    optimal_alpha = list(max_row['alpha'])[0]
    optimal_fit_prior = fit_prior_value
    
    print(results_df)
    return optimal_alpha, optimal_fit_prior

def get_positive_airline(df, airline_col, sentiment_col):
    """Return the airline with the highest percentage of positive sentiment tweets"""
    positive_sentiment = {}

    for airline in np.unique(df[airline_col]):
        airline_tweets = df[df[airline_col] == airline]
        positive_tweet_count = len(airline_tweets[airline_tweets[sentiment_col] == 2])
        positive_sentiment[airline] = np.round(positive_tweet_count/len(airline_tweets),2)

    #get key (airline name) associated with max sentiment value
    positive_airline = [key for key in positive_sentiment.keys() if 
                        positive_sentiment[key] == max(positive_sentiment.values())][0]
    
    return positive_airline

if __name__ == "__main__":
    if len(sys.argv) == 2:
        file_location = sys.argv[1].strip()

        if(os.path.exists(file_location) and file_location.endswith("Tweets.csv")):
            try:
                columns = ['airline_sentiment', 'airline', 'text']
                pd.read_csv(file_location, usecols = columns)
            except ValueError:
                print("Please enter a valid csv with airline_sentiment, airline, and text columns")
        
            target_col = "airline_sentiment"
            normalized_counts, df = preprocess_tweets_df(file_location)
            X_train, X_test, y_train, y_test =  train_test_split(normalized_counts, df[target_col], test_size=0.1)

            # retrieve optimal hyperparameters 
            optimal_alpha, optimal_fit_prior = mnb_optimizer(num_searches = 5)

            # output model with optimal hyperparameters found with optimizer function
            model = MultinomialNB(alpha = optimal_alpha, fit_prior = optimal_fit_prior)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            accuracy = np.round(np.mean(y_pred == y_test), 2)
            print(f"\nOptimal alpha: {optimal_alpha}\nOptimal fit prior: {optimal_fit_prior}\nAccuracy: {accuracy*100}\n")

            # get average sentiment 
            print("Average Sentiment per Airline")
            print(df.groupby('airline').mean())

            #get airline with most positive sentiment tweets
            positive_airline = get_positive_airline(df, 'airline', target_col)
            print(f"\nAirline with most positive sentiment: {positive_airline}")
        else:
            print("Please enter a valid file location")
      