from preprocess import *
from embeddings import *
from modelling.modelling import *
from modelling.data_model import *
import random
import numpy as np

seed = 0
random.seed(seed)
np.random.seed(seed)

def load_data():
    df = get_input_data()
    return df

def preprocess_data(df):
    df = de_duplication(df)
    df = noise_remover(df)
    df[Config.TICKET_SUMMARY] = translate_to_en(df[Config.TICKET_SUMMARY].tolist())
    return df

def get_embeddings(df):
    X = get_tfidf_embd(df)
    return X, df

def get_data_object(X, df):
    return Data(X, df)

def perform_modelling(data, df, name):
    model_predict(data, df, name)

if __name__ == '__main__':
    df = load_data()
    df = preprocess_data(df)
    df[Config.INTERACTION_CONTENT] = df[Config.INTERACTION_CONTENT].values.astype('U')
    df[Config.TICKET_SUMMARY] = df[Config.TICKET_SUMMARY].values.astype('U')

    X, df = get_embeddings(df)
    data = get_data_object(X, df)
    perform_modelling(data, df, 'RandomForest_BaseRun')