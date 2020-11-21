from nltk.tag import StanfordNERTagger
from nltk.tokenize import word_tokenize
from config.creds import authenticate
import preprocessor as p
import schedule
import nltk
import time
import os

numIterations = 0
results = {}


def named_entity_recognition(tweets):
    """named entity recognition for a batch of tweets"""

    base_folder = "../stanford-ner-4.0.0"
    classifier = "english.all.3class.distsim.crf.ser.gz"
    jar_file = "stanford-ner-4.0.0.jar"

    classification_model = base_folder + '/classifiers/' + classifier
    stanford_tagger_jar = base_folder + '/' + jar_file


    st = StanfordNERTagger(classification_model, stanford_tagger_jar,encoding='utf-8')

    target_entities = ['LOCATION', 'PERSON', 'ORGANIZATION']

    for tweet in tweets:
        tokenized_text = word_tokenize(tweet)
        classified_text = st.tag(tokenized_text)

        for text, entity in classified_text:
            if entity in target_entities:
                if text not in results:
                    results[text] = 1
                else:
                    results[text]+=1
        print(tweet)
        print(results)
        print("\n")

    
    return results


def twitter_searcher(topic, twitter_filter, language, count, result_type, pull_date):
    global numIterations

    print(f'Running Iteration # {numIterations+1}')
    print("collecting tweets..")
    API = authenticate("creds.ini")
    query_string = f"{topic} -filter:{twitter_filter}"
    results = API.search(q = query_string,lang=language, result_type = result_type, count = count, until=pull_date)

    #get only the 
    tweets = [p.clean(result.text) for result in results]
    tweets = list(set(tweets))

    numIterations+=1

    named_entity_recognition(tweets)


def scheduleIteration(run_interval, total_iterations): 

    global numIterations 

    twitter_searcher("NBA", "retweets","en", 5, "recent", "2020-11-20")

    schedule.every(run_interval).minutes.do(twitter_searcher, "NBA", "retweets",
    "en", 20, "recent", "2020-11-20")

    while(numIterations < total_iterations):
        schedule.run_pending()
        time.sleep(1)


scheduleIteration(1,2)