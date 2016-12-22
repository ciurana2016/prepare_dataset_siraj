import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def dat_data(poke_data):

    # Load the csv
    df = pd.read_csv(poke_data)

    # Remove #, Name, Type2, Generation and Legendary
    df = df.drop(df.columns[[0, 1, 3, 11, 12]], axis=1)

    # Labeling the data
    data = df.drop('Type 1', 1)
    labels = df['Type 1']

    # Train and test split
    train_data, test_data, train_labels, test_labels = train_test_split(data, labels, test_size=0.2, random_state=1)

    # All the needed data
    response = {
        'train_data' : train_data,
        'test_data'  : test_data,
        'train_labels' : train_labels,
        'test_labels' : test_labels,
    }

    return response