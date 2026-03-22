import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from Config import Config

def get_tfidf_embd(df):
    # Combine Ticket Summary and Interaction Content as input text
    df[Config.INTERACTION_CONTENT] = df[Config.INTERACTION_CONTENT].fillna('')
    df[Config.TICKET_SUMMARY] = df[Config.TICKET_SUMMARY].fillna('')

    text = df[Config.TICKET_SUMMARY] + ' ' + df[Config.INTERACTION_CONTENT]

    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    X = vectorizer.fit_transform(text)

    return X