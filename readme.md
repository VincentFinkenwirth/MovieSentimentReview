
# Movie sentiment analysis
## 1) Introduction
This projects aim is to analyze the sentiment of movie reviews.
To achieve this goal, multiple models are trained with different algorithms and levels of complexity.
The models are trained on a combination of the kaggle datasets, which includes the 50k IMDB movie reviews dataset and the 50k IMDB movie reviews dataset and 50k rows of rotten tomatoes data.
This dataset of 100k is split into training and preliminary testing sets and the remainder of the rotten tomatoes data is used as final test for generalization

## 2) How to use:
1) Clone the repository:
```bash
git clone git clone https://github.com/VincentFinkenwirth/MovieSentimentReview.git
```
2) Check **Implementation.ipynb** for the structure, implementation and results of the project. (Requires jupyter and optionally pandas to run some cells)
3) Install the required packages:
```bash
pip install -r requirements.txt
```
4) Optionally to have torch with cuda support, install pytorch with the appropriate version for your system from https://pytorch.org/get-started/locally/
4) run **initializer.py** to download files, that are to large to be stored on github, from Google drive. (Implementation and processes can be checked without, but functionality will be limited)
```bash
python initializer.py
5) To see model training and finetuning information, run **mlflou ui** in terminal.
```

---
## Overview
## 3) Data
1) Kaggle imdb dataset with 25k positive and negative reviews https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews
2) Rotten tomatoe movie reviews dataset containing over 1mio reviews to have a broad range of genres and review styles to test generalization https://www.kaggle.com/datasets/andrezaza/clapper-massive-rotten-tomatoes-movies-and-reviews?select=rotten_tomatoes_movie_reviews.csv

---

## 4) Models
### 4.1) Vectorizer
For classical and gradient boosting models, a TF-IDF vectorizer and a Count vectorizer are used. The BERT model uses the pre-trained BERT tokenizer to tokenize the text.
### 4.2) Classical
4 different classical models are chosen to compare the performance of different algorithms aswell as their generalization capabilities
1) Logistic regression
2) Random forest
3) Support vector machine
4) Naive bayes

### 4.3) Catboost (Gradient boosting)
Catboost is chosen as a gradient boosting algorithm as it is known for its good performance. Gradient boosting often provides great results at managable computational costs.

### 4.4) BERT for classification (Transformer)
BERT is chosen as a transformer model as it is known for exceptional accuracy in NLP tasks. As the model is pre-trained on a large corpus of text, it is expected to have greater generalization capabilities than the other models.

