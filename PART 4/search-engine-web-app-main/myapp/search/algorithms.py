import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import collections
from collections import defaultdict
import json
from array import array
import math
import numpy as np
from numpy import linalg as la
import pickle
import spacy

from myapp.search.objects import ResultItem, Document
from myapp.search.load_corpus import _build_terms


def search_in_corpus(query, search_id, corpus: dict, index, idf, tf, algorithm = "TF_IDF"):
        
    query = _build_terms(query)
    tweets = []
    
    try:
        term_tweets = [posting[0] for posting in index[query[0]]] # indexes == index[query[0]]
        for t_id in term_tweets:
            intersection = set(corpus[t_id].processed_tweet).intersection(set(query))
            if set(query) == intersection:
                tweets.append(t_id)

    except:
        pass

    res = []
    if algorithm == "TF_IDF":
        result_tweets, tweet_scores = rank_documents_TF_IDF(query, tweets, index, idf, tf)
    elif algorithm == "CUSTOM":
        result_tweets, tweet_scores = rank_documents_with_custom_score(query, tweets, index, corpus)
    elif algorithm == "Tweet2Vec":
        result_tweets, tweet_scores = rank_documents_tweet2vec(query, tweets, index, corpus)

    for result in result_tweets:
        item: ResultItem = corpus[result]
        res.append(ResultItem(item.id, item.title, item.description, item.doc_date,
                              "doc_details?id={}&search_id={}&param2=2".format(item.id, search_id), tweet_scores))    
    return res

def create_index_tfidf(lines, num_tweets):

    index = defaultdict(list)
    tf = defaultdict(list)  # term frequencies of terms in documents (documents in the same order as in the main index)
    df = defaultdict(int)  # document frequencies of terms in the corpus
    url_index = defaultdict(str)
    idf = defaultdict(float)

    for line in lines:
        
        terms = line.processed_tweet #tweet_information[tweet_id]['Processed Tweet']
        url_index[line.id] = line.url #tweet_information[tweet_id]['Tweet_url']

        current_page_index = {}

        for position, term in enumerate(terms):
            try:
                current_page_index[term][1].append(position)
            except:
                current_page_index[term] = [line.id, array('I',[position])] #'I' indicates unsigned int (int in Python)

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
            idf[term] = np.round(np.log(float(num_tweets/df[term])), 4)

    return index, tf, df, idf, url_index

def rank_documents_TF_IDF(terms, tweets, index, idf, tf):
    tweet_vectors = defaultdict(lambda: [0] * len(terms)) # I call doc_vectors[k] for a nonexistent key k, the key-value pair (k,[0]*len(terms)) will be automatically added to the dictionary
    query_vector = [0] * len(terms)

    query_terms_count = collections.Counter(terms)  # get the frequency of each term in the query.
    
    query_norm = la.norm(list(query_terms_count.values()))

    for termIndex, term in enumerate(terms):  #termIndex is the index of the term in the query
        if term not in index:
            continue

        query_vector[termIndex] = query_terms_count[term]/query_norm * idf[term]

        for tweet_index, (tweet, postings) in enumerate(index[term]):
            if tweet in tweets:
                tweet_vectors[tweet][termIndex] = tf[term][tweet_index] * idf[term]

    # Cosine similarity
    tweet_scores=[[np.dot(curTweetVec, query_vector), tweet] for tweet, curTweetVec in tweet_vectors.items() ]
    tweet_scores.sort(reverse=True)
    result_tweets = [x[1] for x in tweet_scores]

    return result_tweets, tweet_scores

# Function to rank documents with custom score and cosine similarity
def rank_documents_with_custom_score(terms, tweets, index, corpus):
    tweet_vectors = defaultdict(lambda: [0] * len(terms))
    query_vector = [0] * len(terms)

    query_terms_count = collections.Counter(terms)
    query_norm = la.norm(list(query_terms_count.values()))

    for termIndex, term in enumerate(terms):
        if term not in index:
            continue

        query_vector[termIndex] = query_terms_count[term] / query_norm  # Using TF for the query

        for tweet_index, (tweet, postings) in enumerate(index[term]):
            if tweet in tweets:
                # Compute the TF/len(doc) for the term in the document
                tf_value = len([postings[0]])
                my_score = 0.25 * corpus[tweet].likes + 0.75 * corpus[tweet].retweets
                tweet_vectors[tweet][termIndex] = (tf_value / len(corpus[tweet].processed_tweet)) + my_score

    # Cosine similarity
    tweet_scores = [[np.dot(curTweetVec, query_vector), tweet] for tweet, curTweetVec in tweet_vectors.items()]
    tweet_scores.sort(reverse=True)
    result_docs = [x[1] for x in tweet_scores]

    return result_docs, tweet_scores

# Load spaCy model with word embeddings
nlp = spacy.load("en_core_web_md")  # You can replace "en_core_web_md" with other available models

# Function to calculate tweet representation using spaCy's word embeddings
def calculate_tweet_representation(tweet):
    tweet_vector = np.zeros(nlp.vocab.vectors.shape[1])
    word_count = 0

    for word in tweet:
        if nlp.vocab.has_vector(word):
            tweet_vector += nlp.vocab[word].vector
            word_count += 1

    if word_count > 0:
        tweet_vector /= word_count

    return tweet_vector

# Function to rank documents using spaCy's word embeddings + cosine similarity
def rank_documents_tweet2vec(terms, tweets, index, corpus):
    tweet_vectors = defaultdict(lambda: np.zeros(nlp.vocab.vectors.shape[1]))
    query_vector = np.zeros(nlp.vocab.vectors.shape[1])

    # Calculate the query vector using spaCy's word embeddings
    for term in terms:
        if nlp.vocab.has_vector(term):
            query_vector += nlp.vocab[term].vector

    # Normalize the query vector
    query_norm = np.linalg.norm(query_vector)
    if query_norm > 0:
        query_vector /= query_norm

    for tweet_index, (tweet, postings) in enumerate(index[term]):
        if tweet in tweets:
            # Calculate the tweet vector for the document using spaCy's word embeddings
            tweet_vector = calculate_tweet_representation(corpus[tweet].processed_tweet)

            # Normalize the tweet vector
            tweet_norm = np.linalg.norm(tweet_vector)
            if tweet_norm > 0:
                tweet_vector /= tweet_norm

            # Cosine similarity
            cosine_similarity = np.dot(tweet_vector, query_vector)

            tweet_vectors[tweet] = cosine_similarity

    # Sort documents based on cosine similarity
    tweet_scores = sorted(tweet_vectors.items(), key=lambda x: x[1], reverse=True)
    result_tweets = [tweet[0] for tweet in tweet_scores]

    return result_tweets, tweet_scores