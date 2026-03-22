import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from model.randomforest import RandomForest
from Config import Config

seed = 0

def get_text_features(df):
    text = df[Config.TICKET_SUMMARY] + ' ' + df[Config.INTERACTION_CONTENT]
    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    X = vectorizer.fit_transform(text)
    return X

def run_level(df, target_col, label):
    X = get_text_features(df)
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed
    )

    class LevelData:
        pass

    data = LevelData()
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

    # Return full predictions for filtering next level
    all_predictions = model.mdl.predict(X)
    return all_predictions

def run_hierarchical_classification(df):
    print("\nLevel 1: Classifying Type 2")
    print("="*60)
    type2_predictions = run_level(df, Config.TYPE_2_COL, 'Level1 - Type 2')

    df = df.copy()
    df['predicted_type2'] = type2_predictions

    # Level 2: for each Type 2 class, filter and classify Type 3
    for type2_class in df[Config.TYPE_2_COL].unique():
        subset = df[df['predicted_type2'] == type2_class]

        if len(subset) < 5:
            print(f"\nSkipping Type 3 for '{type2_class}' — not enough data")
            continue

        print(f"\nLevel 2: Classifying Type 3 | Type 2 = '{type2_class}'")
        print("="*60)
        type3_predictions = run_level(
            subset, Config.TYPE_3_COL,
            f'Level2 - Type2={type2_class} → Type 3'
        )

        subset = subset.copy()
        subset['predicted_type3'] = type3_predictions

        # Level 3: for each Type 3 class, filter and classify Type 4
        for type3_class in subset[Config.TYPE_3_COL].unique():
            subset2 = subset[subset['predicted_type3'] == type3_class]

            if len(subset2) < 5:
                print(f"\nSkipping Type 4 for '{type2_class} → {type3_class}' — not enough data")
                continue

            print(f"\nLevel 3: Classifying Type 4 | Type 2='{type2_class}' → Type 3='{type3_class}'")
            print("="*60)
            run_level(
                subset2, Config.TYPE_4_COL,
                f'Level3 - Type2={type2_class}, Type3={type3_class} → Type 4'
            )