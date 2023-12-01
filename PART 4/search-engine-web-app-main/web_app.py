import os
from json import JSONEncoder
from collections import defaultdict
from collections import Counter
from user_agents import parse
# from geopy.geocoders import Nominatim
import requests
# import geoip2.database

# pip install httpagentparser
import httpagentparser  # for getting the user agent as json
import nltk
import pickle
from flask import Flask, render_template, session
from flask import request
from datetime import datetime, timedelta
import pytz
import numpy as np

from myapp.analytics.analytics_data import AnalyticsData, ClickedDoc
from myapp.search.load_corpus import load_corpus, _build_terms
from myapp.search.objects import Document, StatsDocument
from myapp.search.search_engine import SearchEngine
from myapp.search.algorithms import create_index_tfidf


# *** for using method to_json in objects ***
def _default(self, obj):
    return getattr(obj.__class__, "to_json", _default.default)(obj)


_default.default = JSONEncoder().default
JSONEncoder.default = _default

# end lines ***for using method to_json in objects ***

# instantiate the Flask application
app = Flask(__name__)

# random 'secret_key' is used for persisting data in secure cookie
app.secret_key = 'afgsreg86sr897b6st8b76va8er76fcs6g8d7'
# open browser dev tool to see the cookies
app.session_cookie_name = 'IRWA_SEARCH_ENGINE'

# instantiate our search engine
search_engine = SearchEngine()

# instantiate our in memory persistence
analytics_data = AnalyticsData()

# print("current dir", os.getcwd() + "\n")
# print("__file__", __file__ + "\n")
full_path = os.path.realpath(__file__)
path, filename = os.path.split(full_path)
# print(path + ' --> ' + filename + "\n")
# load documents corpus into memory.
file_path = path + "/Rus_Ukr_war_data.json"#"/Rus_Ukr_war_data_test.json"

# file_path = "../../tweets-data-who.json"
corpus = load_corpus(file_path)
print("loaded corpus. first elem:", list(corpus.values())[0])

ll = list(corpus.values())
try:
    with open('saved_steps.pkl', 'rb') as file:
        data = pickle.load(file)

    idx = data["index"]
    tf = data["tf"]
    df = data["df"]
    idf = data["idf"]
    url_index = data["url_index"]
except:
    idx, tf, df, idf, url_index = create_index_tfidf(ll, len(ll))
    data = {"index": idx, "tf": tf, "df": df, "idf": idf, "url_index": url_index}
    with open('saved_steps.pkl', 'wb') as file:
        pickle.dump(data, file)  

# Home URL "/"
@app.route('/')
def index():
    print("starting home url /...")

    # flask server creates a session by persisting a cookie in the user's browser.
    # the 'session' object keeps data between multiple requests
    session['some_var'] = "IRWA 2023 home"

    user_agent = request.headers.get('User-Agent')
    print("Raw user browser:", user_agent)

    user_ip = request.remote_addr
    agent = httpagentparser.detect(user_agent)

    print("Remote IP: {} - JSON user browser {}".format(user_ip, agent))

    print(session)

    return render_template('index.html', page_title="Welcome")


@app.route('/search', methods=['POST'])
def search_form_post():
    search_query = request.form['search-query']
    selected_algorithm = request.form['algorithm'] # Default to TF_IDF if not selected

    session['last_search_query'] = search_query
    session['last_algorithm'] = selected_algorithm

    search_id = analytics_data.save_query_terms(search_query)

    results = search_engine.search(search_query, search_id, corpus, idx, idf, tf, selected_algorithm)

    found_count = len(results)
    session['last_found_count'] = found_count

    print(session)

    return render_template('results.html', results_list=results, page_title="Results", query=search_query, found_counter=found_count, algorithm=selected_algorithm)

@app.route('/doc_details', methods=['GET'])
def doc_details():
    # getting request parameters:
    # user = request.args.get('user')

    print("doc details session: ")
    print(session)

    res = session["some_var"]

    print("recovered var from session:", res)

    # get the query string parameters from request
    clicked_doc_id = request.args["id"]
    algorithm = session['last_algorithm']
    query = session['last_search_query']
    query_terms = _build_terms(session['last_search_query'])

    user_agent = request.headers.get('User-Agent')
    agent = httpagentparser.detect(user_agent)
    browser = agent['browser']['name']
    platform = agent['platform']['name']
    os = agent['os']['name']
    bot = agent['bot']
    user_ip = request.remote_addr

    doc = corpus[int(clicked_doc_id)]
    print("click in id={}".format(clicked_doc_id))

    def check_in(dictionary, subkey):
        if subkey in dictionary.keys(): dictionary[subkey] += 1
        else: dictionary[subkey] = 1

    # store data in statistics table 1
    if clicked_doc_id in analytics_data.fact_clicks.keys(): check_in(analytics_data.fact_clicks[clicked_doc_id], query)
    else: analytics_data.fact_clicks[clicked_doc_id] = {query: 1}

    if clicked_doc_id in analytics_data.fact_clicks_stats.keys(): check_in(analytics_data.fact_clicks_stats[clicked_doc_id], query)
    else: analytics_data.fact_clicks_stats[clicked_doc_id] = {query: 1}

    total_sum = sum(analytics_data.fact_clicks.get(clicked_doc_id, {}).values())

    print("fact_clicks count for id={} is {}".format(clicked_doc_id, total_sum))
    
    # store data in statistics table 2
    check_in(analytics_data.fact_algorithms, algorithm)
    check_in(analytics_data.fact_algorithms_stats, algorithm)


    print("fact_algorithms count for algorithm={} is {}".format(algorithm, analytics_data.fact_algorithms[algorithm]))

    # store data in statistics table 3
    for term in query_terms:
        check_in(analytics_data.fact_terms, term)
        check_in(analytics_data.fact_terms_stats, term)
    
    print("fact_terms count for term={} is {}".format(term, analytics_data.fact_terms[term]))


    # store data in statistics table 4
    if 'platforms' in analytics_data.fact_agents.keys(): check_in(analytics_data.fact_agents['platforms'], platform)
    else: analytics_data.fact_agents['platforms'] = {platform: 1}

    if 'oss' in analytics_data.fact_agents.keys(): check_in(analytics_data.fact_agents['oss'], os)
    else: analytics_data.fact_agents['oss'] = {os: 1} 

    if 'browsers' in analytics_data.fact_agents.keys(): check_in(analytics_data.fact_agents['browsers'], browser)
    else: analytics_data.fact_agents['browsers'] = {browser: 1} 

    if 'bot' in analytics_data.fact_agents.keys(): check_in(analytics_data.fact_agents['bot'], bot)
    else: analytics_data.fact_agents['bot'] = {bot: 1}

    if 'platforms' in analytics_data.fact_agents_stats.keys(): check_in(analytics_data.fact_agents_stats['platforms'], platform)
    else: analytics_data.fact_agents_stats['platforms'] = {platform: 1}

    if 'oss' in analytics_data.fact_agents_stats.keys(): check_in(analytics_data.fact_agents_stats['oss'], os)
    else: analytics_data.fact_agents_stats['oss'] = {os: 1} 

    if 'browsers' in analytics_data.fact_agents_stats.keys(): check_in(analytics_data.fact_agents_stats['browsers'], browser)
    else: analytics_data.fact_agents_stats['browsers'] = {browser: 1} 

    if 'bot' in analytics_data.fact_agents_stats.keys(): check_in(analytics_data.fact_agents_stats['bot'], bot)
    else: analytics_data.fact_agents_stats['bot'] = {bot: 1}


    # store data in statistics table 5
    check_in(analytics_data.fact_ip, user_ip)
    check_in(analytics_data.fact_ip_stats, user_ip)

    print("fact_ip count for term={} is {}".format(browser, analytics_data.fact_ip[user_ip]))

    # store data in statistics table 6
    terms_per_query = len(query_terms)
    check_in(analytics_data.fact_query_sizes, terms_per_query)
    check_in(analytics_data.fact_query_sizes_stats, terms_per_query)

    print("fact_query_sizes count for size={} is {}".format(terms_per_query, analytics_data.fact_query_sizes[terms_per_query]))

    # store data in statistics table 7
    response = requests.get(f"https://ipinfo.io/{user_ip}/json")
    data = response.json()
    visitor_city = data.get("city", "Unknown (private or local address)")
    check_in(analytics_data.fact_city, visitor_city)
    check_in(analytics_data.fact_city_stats, visitor_city)

    print("fact_city count for size={} is {}".format(visitor_city, analytics_data.fact_city[visitor_city]))

    # store data in statistics table 8
    click_timestamp = datetime.now(pytz.utc)
    if 'last_click_timestamp' in session:
        last_click_timestamp = session['last_click_timestamp']
        last_click_timestamp = last_click_timestamp.replace(tzinfo=pytz.utc)

        dwell_time = (click_timestamp - last_click_timestamp).total_seconds()

        check_in(analytics_data.fact_dwell_time, dwell_time)
        analytics_data.fact_dwell_time[dwell_time] = dwell_time

        check_in(analytics_data.fact_dwell_time_stats, dwell_time)
        analytics_data.fact_dwell_time_stats[dwell_time] = dwell_time

        print("Dwell time: {} seconds (bin {})".format(dwell_time, analytics_data.fact_dwell_time))

    session['last_click_timestamp'] = click_timestamp

    # store data in statistics table 9
    interaction_day = click_timestamp.strftime('%A')
    check_in(analytics_data.fact_week_days, interaction_day)
    check_in(analytics_data.fact_week_days_stats, interaction_day)


    print("Day of the week: {} (Count: {})".format(interaction_day, analytics_data.fact_week_days[interaction_day]))

    data = {'fact_clicks': analytics_data.fact_clicks, 'fact_algorithms': analytics_data.fact_algorithms, 
            'fact_terms': analytics_data.fact_terms, 'fact_agents': analytics_data.fact_agents, 'fact_ip': analytics_data.fact_ip, 
            'fact_query_sizes': analytics_data.fact_query_sizes, 'fact_city': analytics_data.fact_city, 
            'fact_dwell_time': analytics_data.fact_dwell_time, 'fact_week_days': analytics_data.fact_week_days}
    
    with open('dashboard_data.pkl', 'wb') as file:
        pickle.dump(data, file) 

    return render_template('doc_details.html', doc=doc, page_title="Tweet information")


@app.route('/stats', methods=['GET'])
def stats():
    visited_docs_stats = []

    for doc_id in analytics_data.fact_clicks_stats.keys():
        d: Document = corpus[int(doc_id)]
        count = sum(analytics_data.fact_clicks_stats.get(doc_id, {}).values())
        doc = ClickedDoc(doc_id, d.description, count)
        visited_docs_stats.append(doc)

    # simulate sort by ranking
    visited_docs_stats.sort(key=lambda doc: doc.counter, reverse=True)
    visited_docs_json_stats = [doc.to_json() for doc in visited_docs_stats]
    
    return render_template('stats.html', session=session, visited_docs=visited_docs_json_stats, query_data=analytics_data.fact_algorithms_stats, 
                           term_data=analytics_data.fact_terms_stats, query_docs=analytics_data.fact_clicks_stats, agent_data=analytics_data.fact_agents_stats, 
                           users_ip=analytics_data.fact_ip_stats, query_size_data=analytics_data.fact_query_sizes_stats, cities=analytics_data.fact_city_stats, 
                           dwell_times=analytics_data.fact_dwell_time_stats, week_days=analytics_data.fact_week_days_stats) 


@app.route('/dashboard', methods=['GET'])
def dashboard():
    visited_docs = []

    for doc_id in analytics_data.fact_clicks.keys():
        d: Document = corpus[int(doc_id)]
        count = sum(analytics_data.fact_clicks.get(doc_id, {}).values())
        doc = ClickedDoc(doc_id, d.description, count)
        visited_docs.append(doc)

    # simulate sort by ranking
    visited_docs.sort(key=lambda doc: doc.counter, reverse=True)
    visited_docs_json = [doc.to_json() for doc in visited_docs]
    
    return render_template('dashboard.html', session=session, visited_docs=visited_docs_json, query_data=analytics_data.fact_algorithms, 
                           term_data=analytics_data.fact_terms, query_docs=analytics_data.fact_clicks, agent_data=analytics_data.fact_agents, 
                           users_ip=analytics_data.fact_ip, query_size_data=analytics_data.fact_query_sizes, cities=analytics_data.fact_city, 
                           dwell_times=analytics_data.fact_dwell_time, week_days=analytics_data.fact_week_days) 


@app.route('/sentiment')
def sentiment_form():
    return render_template('sentiment.html')


@app.route('/sentiment', methods=['POST'])
def sentiment_form_post():
    text = request.form['text']
    nltk.download('vader_lexicon')
    from nltk.sentiment.vader import SentimentIntensityAnalyzer
    sid = SentimentIntensityAnalyzer()
    score = ((sid.polarity_scores(str(text)))['compound'])
    return render_template('sentiment.html', score=score)


if __name__ == "__main__":
    app.run(port=8088, host="0.0.0.0", threaded=False, debug=True)
