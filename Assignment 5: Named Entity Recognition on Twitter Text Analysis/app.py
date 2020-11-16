from flask import Flask, redirect, url_for, render_template, request
import datetime as dt
from data_collection.collect_tweets import twitter_searcher


app = Flask(__name__)

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

    # retrieve tweets with required parameters 
    results = twitter_searcher(topic, twitter_filter, language, count, result_type, pull_date)
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
  return redirect(url_for("home"))


@app.route("/contact/", methods = ['POST', 'GET'])
def contact():  
  if request.method == 'POST':
    return redirect(url_for("home"))
  else:
    return render_template("contact.html")

if __name__ == "__main__":
  app.run(debug=True)
