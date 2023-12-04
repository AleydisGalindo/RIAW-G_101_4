# Part4: User Interface and Web Analytics

## Overview

We have used the simple Python web framework Flask that runs its own development web server to build a search engine to process and rank tweets related to the Russo-Ukrainian war. The code is organized into several parts, each serving a specific purpose.

Firstly we load the corpus from JSON files, transforming it into a DataFrame, and apply various data cleaning techniques. Then, we call our text search and ranking algorithms which include TF-IDF ranking, custom scoring, and word2vec-based ranking on a corpus of tweets.
Given our results we then worked on linking all of the `.html` templates together and displaying our statistics through a series of graphs.

## Dependencies

- **Python:** This script requires Python 3.10.12 to run.
- **Libraries:** json, datetime, Faker, random, re, nltk, collections, defaultdict, array, math, numpy, spacy, pandas

Ensure the required libraries are installed using appropriate commands, such as pip install nltk and pip install numpy.

# Download the spaCy model with word embeddings
```bash
python3 -m spacy download en_core_web_md
```

## How to Run

1. **Data Files:** Ensure that you have the required data files in the specified locations.
   - `Rus_Ukr_war_data.json`: JSON file containing tweet data.

2. **Run the Script:** Execute the script in a Python environment typing the console command:
```bash
`python3 search-engine-web-app-main/web_app.py`
```

## Console Output

- Relevant information on the actions taken on the development web server and its input.

## Additional Notes

- The script is designed for educational purposes and involves search algorithms applied to a tweet corpus.
- Ensure the correct paths to data files are provided in the script.