# visualization.py

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from preprocessing import load_data

def basic_eda(df: pd.DataFrame):
    """
    Perform basic exploratory data analysis on the SMS dataset.
    """
    print("Data shape:", df.shape)
    print("\nColumns:", df.columns.tolist())
    print("\nFirst few rows:\n", df.head())
    print("\nMissing values:\n", df.isnull().sum())

def plot_class_distribution(df: pd.DataFrame):
    """
    Plot the distribution of spam vs ham.
    """
    plt.figure(figsize=(6,4))
    sns.countplot(x='label', data=df, palette='viridis')
    plt.title("SMS Spam vs Ham Distribution")
    plt.xlabel("Message Type (spam or ham)")
    plt.ylabel("Count")
    plt.show()

def visualize_data(file_path: str):
    """
    Load the SMS data and run visualizations.
    """
    df = load_data(file_path)
    basic_eda(df)
    plot_class_distribution(df)

if __name__ == "__main__":
    file_path = "../spam.csv"  # Update this path as needed
    visualize_data(file_path)
