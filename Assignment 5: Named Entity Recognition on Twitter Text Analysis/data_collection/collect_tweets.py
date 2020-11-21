""" 

"""

from config.creds import authenticate
import preprocessor as p


def twitter_searcher(topic, twitter_filter, language, count, result_type, pull_date): 
  print("collecting tweets..")
  API = authenticate("creds.ini")
  query_string = f"{topic} -filter:{twitter_filter}"
  results = API.search(q = query_string,lang=language, result_type = result_type, count = count, until=pull_date)

  #get only the 
  tweets = [p.clean(result.text) for result in results]
  tweets = list(set(tweets))

  return tweets


