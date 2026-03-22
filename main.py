from preprocess import *
from embeddings import *
from modelling.modelling import *
from modelling.data_model import *
from design_choice_1.chained_classifier import run_chained_classification
from design_choice_2.hierarchical_classifier import run_hierarchical_classification
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
    # Step 1: Load and preprocess
    df = load_data()
    df = preprocess_data(df)
    df[Config.INTERACTION_CONTENT] = df[Config.INTERACTION_CONTENT].values.astype('U')
    df[Config.TICKET_SUMMARY] = df[Config.TICKET_SUMMARY].values.astype('U')

    # Step 2: Base classification (Type 2 only)
    X, df = get_embeddings(df)
    data = get_data_object(X, df)
    perform_modelling(data, df, 'RandomForest_BaseRun')

    # Step 3: Design Choice 1 - Chained Multi-Output
    print("\n" + "="*60)
    print("DESIGN CHOICE 1 - CHAINED MULTI-OUTPUT CLASSIFICATION")
    print("="*60)
    run_chained_classification(df)

    # Step 4: Design Choice 2 - Hierarchical Modelling
    print("\n" + "="*60)
    print("DESIGN CHOICE 2 - HIERARCHICAL MODELLING")
    print("="*60)
    run_hierarchical_classification(df)