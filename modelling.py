# modeling.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import itertools

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB

from preprocessing import preprocess_data
from visualization import visualize_data

def plot_confusion_matrix(cm, classes, title='Confusion Matrix', cmap=plt.cm.Blues):
    """
    Plot a confusion matrix.
    """
    plt.figure(figsize=(6,5))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], 'd'),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.show()

def train_and_evaluate(X, y):
    """
    Train SMS spam classification models and evaluate them.
    Two sample models are included: Logistic Regression and Multinomial Naive Bayes.
    """
    # Split data into training and test sets.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Define pipelines for both models.
    pipelines = {
        "Logistic Regression": Pipeline([
            ('tfidf', TfidfVectorizer()),
            ('clf', LogisticRegression(max_iter=1000))
        ]),
        "Multinomial NB": Pipeline([
            ('tfidf', TfidfVectorizer()),
            ('clf', MultinomialNB())
        ])
    }
    
    for name, pipeline in pipelines.items():
        print(f"\nTraining {name}...")
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        
        print(f"\n{name} Classification Report:")
        print(classification_report(y_test, y_pred))
        
        cm = confusion_matrix(y_test, y_pred)
        plot_confusion_matrix(cm, classes=["ham", "spam"], title=f"{name} Confusion Matrix")
        
def main():
    """
    Main function to visualize data, preprocess it, and train/evaluate SMS spam models.
    """
    file_path = "../spam.csv"  # Update this path as needed

    # Visualize data
    visualize_data(file_path)
    
    # Preprocess data
    X, y = preprocess_data(file_path)
    
    # Train and evaluate models
    train_and_evaluate(X, y)
    
if __name__ == "__main__":
    main()
