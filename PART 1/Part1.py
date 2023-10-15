import csv
import json
from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import re

# Load the JSON data
with open('IRWA_data_2023/Rus_Ukr_war_data.json', 'r') as file:
    ctr = 0

    tweet_document_ids_map = {}

    with open('IRWA_data_2023/Rus_Ukr_war_data_ids.csv', 'r') as map_file:
        doc = csv.reader(map_file, delimiter='\t')
        for row in doc:
            doc_id, tweet_id = row
            tweet_document_ids_map[tweet_id] = doc_id

        for line in file:
            tweet_data = json.loads(line)

            # Extract relevant information
            tweet_text = tweet_data['full_text']
            tweet_id = tweet_data['id_str']
            tweet_date = tweet_data['created_at']
            hashtags = [hashtag['text'] for hashtag in tweet_data['entities']['hashtags']]
            likes = tweet_data['favorite_count']
            retweets = tweet_data['retweet_count']
            #MIRAR BE TOT LO DELS URL, NO ESTA BE!!!!!!!!!!!!!!
            url_matches = re.findall(r'https://t\.co/[a-zA-Z0-9]+', tweet_text)
            tweet_url = url_matches[-1] if url_matches else ''

            # Tokenization
            tokenizer = TweetTokenizer()
            tokens = tokenizer.tokenize(tweet_text)

            # Removing stop words
            stop_words = set(stopwords.words('english'))
            filtered_tokens = [word for word in tokens if word.lower() not in stop_words]

            # Removing punctuation marks
            filtered_tokens = [re.sub(r'[^\w\s]', '', word) for word in filtered_tokens]

            # Stemming
            stemmer = PorterStemmer()
            stemmed_tokens = [stemmer.stem(word) for word in filtered_tokens]

            # Joining the processed tokens back into a string
            processed_tweet = ' '.join(stemmed_tokens)

            # Map tweet ID to document ID
            doc_id = tweet_document_ids_map.get(tweet_id, f"UnknownDoc_{tweet_id}")

            # Print the results
            if ctr == 9:
                print("Document ID:", doc_id)
                print("Tweet ID:", tweet_id)
                print("Tweet Date:", tweet_date)
                print("Hashtags:", hashtags)
                print("Likes:", likes)
                print("Retweets:", retweets)
                print("Tweet URL:", tweet_url)
                print("Processed Tweet:", processed_tweet)
            ctr += 1
            if ctr > 11:
                break
        

