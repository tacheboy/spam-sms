# preprocessing.py

import pandas as pd
import re
import string
import nltk

# Ensure that the required NLTK resources are downloaded.
nltk.download('stopwords', quiet=True)
from nltk.corpus import stopwords

# Global variables for text cleaning.
STOPWORDS = set(stopwords.words('english'))
PUNCT_REGEX = re.compile(f"[{re.escape(string.punctuation)}]")

def load_data(file_path: str) -> pd.DataFrame:
    """
    Load the CSV file and keep only the first two columns ("v1" and "v2").
    Any extra columns are dropped.
    """
    df = pd.read_csv(file_path, encoding="latin-1")  # encoding may vary
    # Keep only the first two columns.
    df = df[['v1', 'v2']]
    df = df.rename(columns={'v1': 'label', 'v2': 'text'})
    return df

def clean_text(text: str) -> str:
    """
    Clean the input text:
      - Lowercase
      - Remove punctuation
      - Remove stopwords
    """
    text = text.lower()
    text = PUNCT_REGEX.sub(" ", text)
    tokens = text.split()
    tokens = [word for word in tokens if word not in STOPWORDS]
    return " ".join(tokens)

def preprocess_data(file_path: str):
    """
    Load data from file, clean the text messages, and return X (messages) and y (labels).
    """
    df = load_data(file_path)
    # Clean the text column.
    df['text'] = df['text'].apply(clean_text)
    
    # Extract features and labels.
    X = df['text']
    y = df['label']
    return X, y

if __name__ == "__main__":
    file_path = "./spam.csv"  # Update this path as needed
    X, y = preprocess_data(file_path)
    print("Sample preprocessed messages:")
    print(X.head())
    print("\nLabel distribution:")
    print(y.value_counts())
