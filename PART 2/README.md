# Part2: Indexing and Evaluation

## Overview

This Python script ...

## Dependencies

- **Python:** This script requires Python 3.10.12 to run.
- **Libraries:** Make sure you have the necessary libraries installed. You can install them using the following:

# Install NLTK for natural language processing
```bash
pip install nltk
```

# Install Matplotlib for data visualization
```bash
pip install matplotlib
```

# Install WordCloud for generating word clouds
```bash
pip install wordcloud
```

# Install spaCy for advanced natural language processing
```bash
pip install spacy
```

# Download the spaCy English language model
```bash
python -m spacy download en_core_web_sm
```

## How to Run

1. **Data Files:** Ensure that you have the required data files in the specified locations.
   - `Rus_Ukr_war_data.json`: JSON file containing tweet data.
   - `Rus_Ukr_war_data_ids.csv`: CSV file mapping tweet IDs to document IDs.

2. **Run the Script:**
   A Jupyter Notebook consists of cells. Each cell can contain either code or text.
   To run a code cell, click on the cell to select it and press Shift + Enter on your keyboard. Alternatively, you can use the "Run" button in the toolbar.
   This will execute the script and generate visualizations and analyses based on the provided data.

## Output

- **Plots:** The script generates several plots, including a histogram of word count distribution, a word cloud, and a plot showing the number of tweets over time.
- **Printed Analysis:** The script prints analyses such as average sentence length, vocabulary size, and top retweeted and liked tweets.
- **Entity Recognition:** The script uses spaCy to recognize entities in the processed tweets and displays them for the top retweeted and liked tweets.

## Additional Notes

- The script utilizes functions for text processing, emoticon removal, and entity recognition.
- Ensure the correct paths to data files are provided in the script.
