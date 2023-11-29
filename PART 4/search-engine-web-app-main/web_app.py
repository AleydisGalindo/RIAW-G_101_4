import os
from json import JSONEncoder
from collections import defaultdict
from collections import Counter
from user_agents import parse
from geopy.geocoders import Nominatim

# pip install httpagentparser
import httpagentparser  # for getting the user agent as json
import nltk
import pickle
from flask import Flask, render_template, session
from flask import request

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

    return render_template('results.html', results_list=results, page_title="Results", found_counter=found_count)


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
    query_terms = _build_terms(session['last_search_query'])
    user_agent = request.headers.get('User-Agent')
    agent = httpagentparser.detect(user_agent)
    browser = agent['browser']['name']

    doc = corpus[int(clicked_doc_id)]
    p1 = int(request.args["search_id"])  # transform to Integer
    p2 = int(request.args["param2"])  # transform to Integer
    print("click in id={}".format(clicked_doc_id))

    # store data in statistics table 1
    if clicked_doc_id in analytics_data.fact_clicks.keys():
        analytics_data.fact_clicks[clicked_doc_id] += 1
    else:
        analytics_data.fact_clicks[clicked_doc_id] = 1

    print("fact_clicks count for id={} is {}".format(clicked_doc_id, analytics_data.fact_clicks[clicked_doc_id]))

    # store data in statistics table 2
    if algorithm in analytics_data.fact_algorithms.keys():
        analytics_data.fact_algorithms[algorithm] += 1
    else:
        analytics_data.fact_algorithms[algorithm] = 1

    print("fact_algorithms count for algorithm={} is {}".format(algorithm, analytics_data.fact_algorithms[algorithm]))

    # store data in statistics table 3
    for term in query_terms:
        if term in analytics_data.fact_terms.keys():
            analytics_data.fact_terms[term] += 1
        else:
            analytics_data.fact_terms[term] = 1

    print("fact_terms count for term={} is {}".format(term, analytics_data.fact_terms[term]))

    # store data in statistics table 4
    if browser in analytics_data.fact_browsers.keys():
        analytics_data.fact_browsers[browser] += 1
    else:
        analytics_data.fact_browsers[browser] = 1

    print("fact_browser count for term={} is {}".format(browser, analytics_data.fact_browsers[browser]))

    return render_template('doc_details.html', doc=doc, page_title="Tweet information")


@app.route('/stats', methods=['GET'])
def stats():
    """
    Show simple statistics example. ### Replace with dashboard ###
    :return:
    """

    docs = []
    # ### Start replace with your code ###

    for doc_id in analytics_data.fact_clicks:
        row: Document = corpus[int(doc_id)]
        count = analytics_data.fact_clicks[doc_id]
        doc = StatsDocument(row.id, row.title, row.description, row.doc_date, row.url, count)
        docs.append(doc)

    # simulate sort by ranking
    docs.sort(key=lambda doc: doc.count, reverse=True)
    return render_template('stats.html', clicks_data=docs)
    # ### End replace with your code ###


@app.route('/dashboard', methods=['GET'])
def dashboard():
    visited_docs = []
    print(analytics_data.fact_clicks.keys())
    print(analytics_data.fact_algorithms.keys())
    print(analytics_data.fact_terms.keys())
    for doc_id in analytics_data.fact_clicks.keys():
        d: Document = corpus[int(doc_id)]
        doc = ClickedDoc(doc_id, d.description, analytics_data.fact_clicks[doc_id])
        visited_docs.append(doc)

    # simulate sort by ranking
    visited_docs.sort(key=lambda doc: doc.counter, reverse=True)
    visited_docs_json = [doc.to_json() for doc in visited_docs]
    

    #Visitor's city from IP
    # user_ip = request.remote_addr
    # geolocator = Nominatim(user_agent="web_app")
    # location = geolocator.geocode(user_ip)
    # visitor_city = location.address.split(",")[-3] if location else "Unknown"

    # for doc in visited_docs: print(doc)
    return render_template('dashboard.html', visited_docs=visited_docs_json, preferred_browser=analytics_data.fact_browsers, query_data=analytics_data.fact_algorithms, term_data=analytics_data.fact_terms) #visitor_city


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
