# Part3: Ranking

## Overview

This Python script consists of three main sections. The first one performs TF-IDF ranking on a collection of tweets related to the Russo-Ukrainian war. The second one performs the ranking of the same collection using a custom score: (tf/dL) + 0.25*#likes + 0.75*#retweets. The last part performs the ranking by creating an embedding of the tweets and then using word2vec.

## Dependencies

- **Python:** This script requires Python 3.10.12 to run.
- **Libraries:** Make sure you have the necessary libraries installed. You can install them using the following:

# Install NLTK for natural language processing
```bash
pip install nltk
```

# Install Numpy
```bash
pip install numpy
```

# Download the spaCy model with word embeddings
```bash
python3 -m spacy download en_core_web_md
```

## How to Run

1. **Data Files:** Ensure that you have the required data files in the specified locations.
   - `Rus_Ukr_war_data.json`: JSON file containing tweet data.
   - `Rus_Ukr_war_data_ids.csv`: CSV file mapping tweet IDs to document IDs.

2. **Run the Script:**
   A Jupyter Notebook consists of cells. Each cell can contain either code or text.
   To run a code cell, click on the cell to select it and press Shift + Enter on your keyboard. Alternatively, you can use the "Run" button in the toolbar.
   In the rank_documents section, you will be asked to input a query. It will keep asking you for queries, and can only be stopped by inputing "END". 
   
## Output

- **Ranking:** The script generates a top 20 ranking for the inputed queries.

## Additional Notes

- The script utilizes functions for text processing, emoticon removal, link removal, index creation, document ranking, search TF-IDF and word2vec.
- Ensure the correct paths to data files are provided in the script.