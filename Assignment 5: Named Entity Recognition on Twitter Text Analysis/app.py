from flask import Flask, redirect, url_for, render_template, request
import datetime as dt
from data_collection.collect_tweets import twitter_searcher
from flask_sqlalchemy import SQLAlchemy
# from data_collection.ner import named_entity_recognition

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///tweets.db'

#Initialize database
db = SQLAlchemy(app)

#create db model
class Tweets(db.Model):
  id = db.Column(db.Integer, primary_key = True)
  # text = db.Column


@app.route("/home/", methods = ['POST', 'GET'])
@app.route("/", methods = ['POST', 'GET'])
def home():
  if request.method == "POST":

    # extract values from form 
    pull_count = request.form['pullCount']
    pull_frequency = request.form['pullFrequency']

    topic = request.form['topic']
    twitter_filter = request.form['twitterFilter']
    language = request.form['language']
    count = request.form['tweetCount']
    result_type = request.form['resultType']
    pull_date = request.form['date']


    print(dict(request.form))
    # retrieve tweets with required parameters 
    results = twitter_searcher(topic, twitter_filter, language, count, result_type, pull_date)

    # ner_dict = named_entity_recognition(results)
    # print(ner_dict)


    return render_template("results.html", results = results)

  else:
    today = dt.date.today()
    week_ago = today - dt.timedelta(days=7)
    today = today.strftime("%Y-%m-%d")
    week_ago = week_ago.strftime("%Y-%m-%d")
    return render_template("index.html", current_date = today, week_ago = week_ago)

@app.route("/about/")
def about():
  return redirect(url_for("home"))

@app.route("/pricing/")
def pricing():
  return render_template("pricing.html")


@app.route("/contact/", methods = ['POST', 'GET'])
def contact():  
  if request.method == 'POST':
    return redirect(url_for("home"))
  else:
    return render_template("contact.html")

if __name__ == "__main__":
  app.run(debug=True)
