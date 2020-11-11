import configparser
import tweepy # python wrapper for twitter API 
import os


def authenticate(config_file):  
    """ authenticate into Twitter API """

    directory_path = os.path.abspath(os.path.dirname(__file__))
    config_file_path = os.path.join(directory_path, config_file)
    
    config = configparser.ConfigParser()
    config.read(config_file_path)

    API_KEY, API_SECRET_KEY, ACCESS_TOKEN, ACCESS_SECRET = dict(config['twitter']).values()

    auth = tweepy.OAuthHandler(API_KEY, API_SECRET_KEY)
    auth.set_access_token(ACCESS_TOKEN, ACCESS_SECRET)
    API = tweepy.API(auth)
    return API

authenticate("creds.ini")