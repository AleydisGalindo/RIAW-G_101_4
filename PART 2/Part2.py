import array
import collections
import csv
import datetime
import json
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import re
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from wordcloud import WordCloud
import spacy
from collections import Counter, defaultdict
import time
from array import array
import math
from numpy import linalg as la

def build_terms(line):

    filtered_line = line.lower() ## Transform in lowercase
    filtered_line = filtered_line.split() ## Tokenize the text to get a list of terms
    filtered_line = [re.sub(r'[^\w\s]', '', word) for word in filtered_line] # Removing non-words and non-whitespaces
    
    # Removing stop words
    stop_words = set(stopwords.words("english"))
    filtered_line = [word for word in filtered_line if word not in stop_words]  ## Eliminate the stopwords 

    # Stemming
    stemmer = PorterStemmer()
    filtered_line = [stemmer.stem(word) for word in filtered_line] ## Perform stemming

    return filtered_line

def remove_emoticons(text):
    # Define a pattern to find all the emoticons
    emoticon_pattern = re.compile("[" u"\U0001F600-\U0001F64F" u"\U0001F300-\U0001F5FF" 
                                  u"\U0001F680-\U0001F6FF" u"\U0001F1E0-\U0001F1FF" 
                                  u"\U00002500-\U00002BEF" u"\U00002702-\U000027B0" 
                                  u"\U000024C2-\U0001F251" u"\U0001f926-\U0001f937" 
                                  u"\U00010000-\U0010ffff" u"\u2640-\u2642" 
                                  u"\u2600-\u2B55" u"\u200d" 
                                  u"\u23cf" u"\u23e9" 
                                  u"\u231a" u"\ufe0f" 
                                  u"\u3030" "]+", re.UNICODE)

    # Replace emoticons with an empty string
    text_without_emoticons = emoticon_pattern.sub('', text)

    return str(text_without_emoticons)

def remove_links(text):
    # Define a pattern to match URLs
    url_pattern = re.compile(r'https?://\S+|www\.\S+')

    # Replace URLs with an empty string
    text_without_links = url_pattern.sub('', text)

    return str(text_without_links)        
        

#### PRE-PROCESS OF THE DOCUMENT
# Load the JSON data

with open('IRWA_data_2023/Rus_Ukr_war_data.json', 'r') as fp:
    lines = fp.readlines()
lines = [l.strip().replace(' +', ' ') for l in lines]

tweet_information = {}
for line in lines:

        tweet_data = json.loads(line)

        # Clean the text
        tweet_text = tweet_data['full_text']
        tweet_text = remove_emoticons(tweet_text)
        tweet_text = remove_links(tweet_text)

        # Extract relevant information
        tweet_id = tweet_data['id_str']
        tweet_date = tweet_data['created_at']
        hashtags = [hashtag['text'] for hashtag in tweet_data['entities']['hashtags']]
        likes = tweet_data['favorite_count']
        retweets = tweet_data['retweet_count'] 
        twitter_username = tweet_data['user']['screen_name']
        tweet_url = f"https://twitter.com/{twitter_username}/status/{tweet_id}"

        processed_tweet = build_terms(tweet_text)

        # Store all the tweet information
        tweet_information[tweet_id] = {
            'Tweet ID': tweet_id,
            'Tweet Text': tweet_text,
            'Processed Tweet': processed_tweet,
            'Tweet Date': tweet_date,
            'Hashtags': hashtags,
            'Likes': likes,
            'Retweets': retweets,
            'Tweet_url': tweet_url
        }

# Map tweet IDs with document IDs for evaluation stage
tweet_document_ids_map = {}
with open('IRWA_data_2023/Rus_Ukr_war_data_ids.csv', 'r') as map_file:
    doc = csv.reader(map_file, delimiter='\t')
    for row in doc:
        doc_id, tweet_id = row
        tweet_document_ids_map[tweet_id] = doc_id

#### INDEXING
def create_index_tfidf(lines, num_documents):

    index = defaultdict(list)
    tf = defaultdict(list)  # term frequencies of terms in documents (documents in the same order as in the main index)
    df = defaultdict(int)  # document frequencies of terms in the corpus
    url_index = defaultdict(str)
    idf = defaultdict(float)

    for line in lines:
        tweet_data = json.loads(line)
        tweet_id = tweet_data['id_str']
        
        doc_id = tweet_document_ids_map[tweet_id]
        terms = tweet_information[tweet_id]['Processed Tweet']
        url_index[doc_id] = tweet_information[tweet_id]['Tweet_url']

        current_page_index = {}

        for position, term in enumerate(terms):
            try:
                current_page_index[term][1].append(position)
            except:
                current_page_index[term] = [doc_id, array('I',[position])] #'I' indicates unsigned int (int in Python)

        norm = 0
        for term, posting in current_page_index.items():
            norm += len(posting[1]) ** 2
        norm = math.sqrt(norm)

        for term, posting in current_page_index.items():
            tf[term].append(np.round(len(posting[1])/norm,4))
            df[term] += 1

        for term_page, posting_page in current_page_index.items():
            index[term_page].append(posting_page)

        for term in df:
            idf[term] = np.round(np.log(float(num_documents/df[term])), 4)

    return index, tf, df, idf, url_index

def rank_documents(terms, docs, index, idf, tf, url_index):
    doc_vectors = defaultdict(lambda: [0] * len(terms)) # I call doc_vectors[k] for a nonexistent key k, the key-value pair (k,[0]*len(terms)) will be automatically added to the dictionary
    query_vector = [0] * len(terms)

    query_terms_count = collections.Counter(terms)  # get the frequency of each term in the query.
    
    query_norm = la.norm(list(query_terms_count.values()))

    for termIndex, term in enumerate(terms):  #termIndex is the index of the term in the query
        if term not in index:
            continue

        query_vector[termIndex] = query_terms_count[term]/query_norm * idf[term]

        for doc_index, (doc, postings) in enumerate(index[term]):
            if doc in docs:
                doc_vectors[doc][termIndex] = tf[term][doc_index] * idf[term]

    # Cosine similarity
    doc_scores=[[np.dot(curDocVec, query_vector), doc] for doc, curDocVec in doc_vectors.items() ]
    doc_scores.sort(reverse=True)
    result_docs = [x[1] for x in doc_scores]
    if len(result_docs) == 0:
        print("No results found, try again")
        query = input()
        docs = search_tf_idf(query, index)
    return result_docs

def search_tf_idf(query, index):
    query = build_terms(query)
    docs = set()
    for term in query:
        try:
            term_docs=[posting[0] for posting in index[term]]
            docs |= set(term_docs)
        except:
            pass
    docs = list(docs)
    ranked_docs = rank_documents(query, docs, index, idf, tf, url_index)
    return ranked_docs
    
start_time = time.time()
num_documents = len(lines)
index, tf, df, idf, url_index = create_index_tfidf(lines, num_documents)
print("Total time to create the index: {} seconds" .format(np.round(time.time() - start_time, 2)))

q = True
while q == True:
    print("\nInsert your query or END to stop(i.e.: presidents visiting Kyiv):\n")
    query = input()
    if query == 'END':
        break
    ranked_docs = search_tf_idf(query, index)
    top = 10

    print("\n======================\nTop {} results out of {} for the searched query:\n".format(top, len(ranked_docs)))
    for d_id in ranked_docs[:top]:
        print(ranked_docs[d_id])
        print("doc_id = {} - tweet_url: {}".format(d_id, url_index[d_id]))


#### EVALUATION 
search_results = pd.read_csv("WeAreTheJudges.csv")

## Precision
def precision_at_k(doc_score, y_score, k=10):
    
    order = np.argsort(y_score)[::-1]
    doc_score = np.take(doc_score, order[:k]) #y_true
    relevant = sum(doc_score == 1)
    return float(relevant)/k

# Check for query 1
for i in range(1,5):
    current_query = i
    current_query_res = search_results[search_results["query_id"] == current_query]

    k = 5
    print("QUERY ", current_query)
    print("==> Precision@{}: {}\n".format(k, precision_at_k(current_query_res["is_relevant"], current_query_res["predicted_relevance"], k)))
    print("\nCheck on the dataset sorted by score:\n")

    current_query_res.sort_values("predicted_relevance", ascending=False).head(k)

# ## Recall

# ## Average Precision
# def avg_precision_at_k(doc_score, y_score, k=10):
#     gtp = sum(doc_score == 1)
#     order = np.argsort(y_score)[::-1]
#     doc_score = np.take(doc_score, order[:k])
#     if gtp == 0:
#         return 0
#     n_relevant_at_i = 0
#     prec_at_i = 0
#     for i in range(len(doc_score)):
#         if doc_score[i] == 1:
#             n_relevant_at_i += 1
#             prec_at_i += n_relevant_at_i / (i + 1)
#     return prec_at_i / gtp

# #1 
# avg_precision_at_k(np.array(current_query_res["is_relevant"]), np.array(current_query_res["predicted_relevance"]), 150)
# #avg_precision_at_k(np.array([1,0,0,1,1,0]), np.array([0.9,0.8,0.7,0.6,0.5, 0.4]),6)

# #2
# # Check with 'average_precision_score' of 'sklearn' library

# from sklearn.metrics import average_precision_score

# k = 150
# temp = current_query_res.sort_values("predicted_relevance", ascending=False).head(k)
# average_precision_score(np.array(temp["is_relevant"]), np.array(temp["predicted_relevance"][:k]))

# ## F1-Score

# ## Mean Average Precision (MAP)
# def map_at_k(search_res, k=10):
#     avp = []
#     for q in search_res['query_id'].unique():  # loop over all query id
#         curr_data = search_res[search_res['query_id'] == q]  # select data for current query
#         avp.append(avg_precision_at_k(np.array(curr_data['is_relevant']), np.array(curr_data['predicted_relevance']), k))  #append average precision for current query
#     return sum(avp) / len(avp), avp  # return mean average precision

# map_k, avp = map_at_k(search_results, 10)
# map_k

# ## Mean Reciprocal Rank (MRR)

# def rr_at_k(doc_score, y_score, k=10):
#     order = np.argsort(y_score)[::-1]  # get the list of indexes of the predicted score sorted in descending order.
#     doc_score = np.take(doc_score, order[:k])  # sort the actual relevance label of the documents based on predicted score(hint: np.take) and take first k.
#     if np.sum(doc_score) == 0:  # if there are not relevant doument return 0
#         return 0
#     return 1 / (np.argmax(doc_score == 1) + 1)  # hint: to get the position of the first relevant document use "np.argmax"

# doc_score = np.array([0, 1, 0, 1, 1])
# score = np.array([0.9, 0.5, 0.6, 0.7, 0.2])
# rr_at_k(doc_score, score, 5)

# ## Normalized Discounted Cumulative Gain (NDCG)
# def dcg_at_k(doc_score, y_score, k=10):
#     order = np.argsort(y_score)[::-1]  # get the list of indexes of the predicted score sorted in descending order.
#     doc_score = np.take(doc_score, order[:k])  # sort the actual relevance label of the documents based on predicted score(hint: np.take) and take first k.
#     gain = 2**doc_score - 1   # Compute gain (use formula 7 above)
#     discounts = np.log2(np.arange(len(doc_score)) + 2)  # Compute denominator
#     return np.sum(gain / discounts)  #return dcg@k

# def ndcg_at_k(doc_score, y_score, k=10):
#     dcg_max = dcg_at_k(doc_score, doc_score, k) # Ideal dcg
#     if not dcg_max:
#         return 0
#     return np.round(dcg_at_k(doc_score, y_score, k), 4) # return ndcg@k

# ndcgs = []
# k = 10
# for q in search_results['query_id'].unique(): # loop over all query ids
#     labels = np.array(search_results[search_results['query_id'] == q]["doc_score"]) ## get labels for current query
#     scores = np.array(search_results[search_results['query_id'] == q]["predicted_relevance"]) # get predicted score for current query
#     ndcgs.append(ndcg_at_k(labels, scores, k)) # append NDCG for current query

# avg_ndcg = np.round(sum(ndcgs) / len(ndcgs), 4) # Compute average NDCG
# print("Average ndcg@{}: {}".format(k, avg_ndcg))