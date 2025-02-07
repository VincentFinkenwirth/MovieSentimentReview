
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
2) Check **Implementation.ipynb** for the structure, implementation and results of the project.
3) Install the required packages:
```bash
pip install -r requirements.txt
```
4) Optionally to have torch with cuda support, install pytorch with the appropriate version for your system from https://pytorch.org/get-started/locally/
4) run **initializer.py** to download files to large to be stored on github. (Implementation and processes can be checked without, but functionality will be limited)
```bash
python initializer.py
```

---
## Overview
## 3) Data
1) Kaggle imdb dataset with 25k positive and negative reviews https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews
2) Rotten tomatoe movie reviews dataset containing over 1mio reviews to have a broad range of genres and review styles to test generalization https://www.kaggle.com/datasets/andrezaza/clapper-massive-rotten-tomatoes-movies-and-reviews?select=rotten_tomatoes_movie_reviews.csv
3) For faster training and consistent splits the data is presplit and turned into X_train, X_val, X_test, y_train, y_val, y_test and saved in a directory -> Models load preprocessed data instead of redoing preprocessing during every training round.
### 3.1) Data preprocessing
1) Data preprocessing is done by removing html tags, special characters, stopwords and lemmatizing the words - This code can be found in tools/data_preprocess.py
2) A balanced split of 25k positive and 25k negative reviews is removed from the rotten tomatoes dataset and added to the imdb data to create the training, validation and testing set foundation - This code can be found in tools/data_combination.py

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

---

## 5) Approach
Found in the models folder
Both classical and gradient boosting models are trained using cross validation, while the BERT model uses a more traditional train-validation-test split (due to computational costs).
1) Data preprocessing and splitting into training, validation and test sets (initially only IMDB data)
2) Train classical models using Optuna hyperparameter optimization (including vectorizer parameters)
3) Train Catboost model using Optuna hyperparameter optimization (including vectorizer parameters)
4) Train BERT model using the pre-trained BERT tokenizer and the pre-trained BERT model
5) If possible models undergo hyperparameter optimization using Optuna on the augmented IMDB+Rotten data
5) For models where this is compuationally not feasible, the best performing models found in IMDB hyperparameter tuning are retrained on augmented data.
6) Final model performance is evaluated on the Rotten tomatoes test set
