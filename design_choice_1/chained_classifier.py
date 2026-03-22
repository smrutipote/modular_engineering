import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from model.randomforest import RandomForest
from Config import Config

seed = 0

def get_chained_data(df):
    # Build three progressively combined target labels
    df['y_chain1'] = df[Config.TYPE_2_COL]
    df['y_chain2'] = df[Config.TYPE_2_COL] + Config.CHAIN_SEPARATOR + df[Config.TYPE_3_COL]
    df['y_chain3'] = df[Config.TYPE_2_COL] + Config.CHAIN_SEPARATOR + df[Config.TYPE_3_COL] + Config.CHAIN_SEPARATOR + df[Config.TYPE_4_COL]
    return df

def run_chained_classification(df):
    df = get_chained_data(df)

    # Build TF-IDF features
    text = df[Config.TICKET_SUMMARY] + ' ' + df[Config.INTERACTION_CONTENT]
    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    X = vectorizer.fit_transform(text)

    chains = [
        ('y_chain1', 'Type 2'),
        ('y_chain2', 'Type 2 + Type 3'),
        ('y_chain3', 'Type 2 + Type 3 + Type 4'),
    ]

    for col, label in chains:
        print(f"\n{'='*60}")
        print(f"Chain: {label}")
        print(f"{'='*60}")

        y = df[col]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=seed
        )

        # Simple data wrapper to match RandomForest interface
        class ChainData:
            pass

        data = ChainData()
        data.X_train = X_train
        data.X_test = X_test
        data.y_train = y_train
        data.y_test = y_test

        data.get_X_train = lambda d=data: d.X_train
        data.get_X_test = lambda d=data: d.X_test
        data.get_type_y_train = lambda d=data: d.y_train
        data.get_type_y_test = lambda d=data: d.y_test
        data.get_embeddings = lambda: X

        model = RandomForest(label, X, y)
        model.train(data)
        model.predict(X_test)
        model.print_results(data)