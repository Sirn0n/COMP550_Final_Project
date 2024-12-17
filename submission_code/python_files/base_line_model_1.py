# -*- coding: utf-8 -*-
"""COMP550_Assignment1.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1zYgqv_8lhQpiG2ToAIBtARFMkO_-PtVe
"""

from google.colab import drive
drive.mount('/content/drive')

"""## Libraries and Modules"""

# Data acquisiton and representation
import pandas as pd
import numpy as np

# machine learning models related
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, precision_recall_curve
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

# data preprocessing related:
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer # we will use this to apply stemming to our words

# I will try using stemming, and then on another trial I will try using lemmatization to see the difference in the results
from nltk.stem import WordNetLemmatizer
nltk.download('wordnet')
lemmatizer = WordNetLemmatizer() # This is for lemmantization

import re

# Visualization
import matplotlib.pyplot as plt
from wordcloud import WordCloud

"""## Helper methods

## Dataset
"""

training = pd.read_csv("/content/drive/MyDrive/FALL2024/comp550/final_project/sarcasm/reddit/no_context/training_response.csv")
testing = pd.read_csv("/content/drive/MyDrive/FALL2024/comp550/final_project/sarcasm/reddit/no_context/testing_response.csv")

print(training.head())

print(testing.head())

"""# Data preprocessing

## Basic cleaning and data preprocessing decisions:
"""

def to_lowercase(response):
    return response.lower() # Convert to lowercase

def remove_stopwords(response_array):
    '''
    Use split() on the response to convert to array form
    '''
    stop_words = set(stopwords.words('english'))
    return [word for word in response_array if not word in stop_words] # Remove stopwords

def do_clean(dataset_frame):
    for response in dataset_frame['response']:
        response = to_lowercase(response)
        response = response.split()
        response = remove_stopwords(response)
        response = ' '.join(response)
        yield response

"""## Creating Bag-of-Words"""

# we will test the model accuracy depending on whether our bag of words consider unigrams, bigrams and trigrams or not.
def create_bag_of_words(cleaned_corpus, ngram_range=(1,1)):
    count_vectorizer = CountVectorizer(ngram_range=ngram_range, max_features=1600)
    matrix_of_words = count_vectorizer.fit_transform(cleaned_corpus).toarray()
    return matrix_of_words

"""# Defining the classification machine learning model: Logitstic Regression

"""

def preprocess_and_train_models(dataset_raw, truth_status,ngram_range=(1,1)):
    '''
    Preprocess the data (without stemming/lemmatization) and train the models.

    Parameters:
    - dataset_raw: The original dataset containing text data
    - truth_status: The labels for classification
    - ngram_range: N-gram range for CountVectorizer (default is unigram (1,1))

    Returns:
    - A dictionary containing accuracy, confusion matrix, classification report, predicted probabilities (y_proba),
      and city distributions in the training and testing sets.
    '''

    # Step 1: Clean the corpus
    cleaned_corpus = do_clean(dataset_raw)

    # Step 2: Create the Bag of Words model with the chosen n-gram range
    matrix_of_words = create_bag_of_words(cleaned_corpus, ngram_range)

    # Step 3: Split the data into training and testing sets

    training_x, testing_x, training_y, testing_y= train_test_split(
        matrix_of_words, truth_status, test_size=0.20, random_state=34)

    # Step 5: Train and evaluate the classifiers
    classifiers = {"Logistic Regression": LogisticRegression()}

    # Dictionary to store the results (accuracy, confusion matrix, classification report, probabilities)
    results = {}

    # Loop through classifiers and evaluate them
    for name, classifier in classifiers.items():
        # print(f"\nTraining and evaluating {name} with stem_or_lemma={stem_or_lemma} and ngram_range={ngram_range}")
        classifier.fit(training_x, training_y)
        y_pred = classifier.predict(testing_x)

        # Get predicted probabilities for precision-recall curve
        y_proba = classifier.predict_proba(testing_x)[:, 1]  # Get probabilities for the positive class (1)

        # Calculate accuracy
        accuracy = accuracy_score(testing_y, y_pred)

        # Generate confusion matrix and classification report
        conf_matrix = confusion_matrix(testing_y, y_pred)
        class_report = classification_report(testing_y, y_pred, output_dict=True)

        # Store the results for this classifier
        results[name] = {
            'accuracy': accuracy,
            'confusion_matrix': conf_matrix,
            'classification_report': class_report,
        }

    return results

"""# Running the experiment"""

# Define combinations of settings that we mentined above.
combinations = [
    {'stem_or_lemma': None, 'ngram_range': (1,1)},  # unigram
    {'stem_or_lemma': None, 'ngram_range': (1,2)},  # unigram + bigram
    {'stem_or_lemma': None, 'ngram_range': (1,3)},  # unigram + bigram + trigram
]
combined_df = pd.concat([training, testing], axis=0)
# Dataset and labels (truth_status)
truth_status = combined_df.iloc[:, -1].values

# List to store the results for each combination
all_results = []

# Loop through all combinations and evaluate models
for combination in combinations:
    results = preprocess_and_train_models(
        dataset_raw=combined_df,
        truth_status=truth_status,
        ngram_range=combination['ngram_range']
    )

    # Append the results for this combination
    all_results.append({
        'combination': f"{combination['stem_or_lemma'] or 'none'}_{combination['ngram_range']}",
        'results': results
    })

"""## Accessing the results"""

print(all_results)

"""## Visualizing the results."""

def plot_accuracies(results):
    '''
    Plots the accuracy for each classifier across different preprocessing configurations
    and shows the best achieved accuracy for each model.

    Parameters:
    - results: list of dictionaries containing combination names and accuracies
    '''
    classifiers = ["Logistic Regression"]

    # Create a figure for the plot
    fig, ax = plt.subplots(figsize=(14, 8))  # Slightly larger figure for readability

    # Loop through classifiers to create subplots for each
    for classifier in classifiers:
        # Extract the accuracies for the current classifier across all combinations
        accuracies = [result['results'][classifier]['accuracy'] for result in results]
        # Extract the corresponding combinations (labels for the x-axis)
        combinations = [result['combination'] for result in results]

        # Plot the accuracies for each classifier with smaller markers
        ax.plot(combinations, accuracies, label=classifier, marker='o', markersize=6)

        # Find the best accuracy and its corresponding combination
        best_accuracy = max(accuracies)
        best_index = accuracies.index(best_accuracy)
        best_combination = combinations[best_index]

        # Annotate the plot with the best accuracy
        ax.annotate(f'Best: {best_accuracy:.4f}',
                    xy=(best_combination, best_accuracy),
                    xytext=(best_combination, best_accuracy + 0.02),  # Space out the annotations more
                    arrowprops=dict(facecolor='black', shrink=0.05),
                    ha='center')

    ax.set_xlabel('Preprocessing + N-Gram Combinations', fontsize=12)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title('Classifier Performance Across Different Preprocessing Configurations', fontsize=14)
    ax.legend(loc='upper right', fontsize=10)  # Move the legend to avoid overlapping with the plot

    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45, ha='right')

    # Adjust the y-axis range for better spacing
    ax.set_ylim([0.40,0.65])

    # Show the plot with improved layout
    plt.tight_layout()
    plt.show()


plot_accuracies(all_results)  # Pass in the list of results generated from your evaluations