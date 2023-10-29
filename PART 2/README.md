# Part2: Indexing and Evaluation

## Overview

This Python script consists of two main secrions. The first one performs indexing and TF-IDF ranking on a collection of tweets related to the Russo-Ukrainian war. And the second one evaluates the retrieval system previously mentioned. Different evaluation techniques are being implemented, such as: precision@k, recall@k, f1-score@k, etc. Finally there are multiple representations of the tweets through the T-SNE algorithm. 

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

# Install Scikit-learn for TfidfVectorizer and TSNE
```bash
pip install scikit-learn
```

# Install Numpy
```bash
pip install numpy
```

# Install Pandas to interact with the csv files
```bash
pip install pandas
```

## How to Run

1. **Data Files:** Ensure that you have the required data files in the specified locations.
   - `Rus_Ukr_war_data.json`: JSON file containing tweet data.
   - `Rus_Ukr_war_data_ids.csv`: CSV file mapping tweet IDs to document IDs.
   - `Evaluation_gt.csv`: CSV file with the ground truths for the provided queries.
   - `WeAreTheJudges.csv`: CSV file with the ground truths for our selected queries.

2. **Run the Script:**
   A Jupyter Notebook consists of cells. Each cell can contain either code or text.
   To run a code cell, click on the cell to select it and press Shift + Enter on your keyboard. Alternatively, you can use the "Run" button in the toolbar.
   This will execute the script and generate visualizations and analyses based on the provided data.
   In the rank_documents section, you will be asked to input a query. It will keep asking you for queries, and can only be stopped by inputing "END". 
   When running the EVALUATION part of the code, run only the cell with the desired CSV file. This way, all the evaluation techniques results and the T-SNE plots will be representative of that file.

## Output

- **Ranking:** The script generates a top 10 ranking for the inputed queries.
- **Printed Evaluation Techniques:** The script prints results for each evaluation technique such as: precision@k (P@K), recall@k (R@K), average precision@k (AP@K), f1-score@k (F1-Score@K), mean average precision (MAP), mean reciprocal rank (MRR) and normalized discounted cumulative gain (NDCG).
- **Plots:** The script generates several plots following the T-SNE algorithm.

## Additional Notes

- The script utilizes functions for text processing, emoticon removal, link removal, index creation, document ranking, search TF-IDF and evaluation techniques.
- Ensure the correct paths to data files are provided in the script.