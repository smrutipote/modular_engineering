import pandas as pd
import numpy as np
from Config import Config

def get_input_data():
    # Load both CSV files and combine them
    df1 = pd.read_csv('data/AppGallery.csv')
    df2 = pd.read_csv('data/Purchasing.csv')
    df = pd.concat([df1, df2], ignore_index=True)
    return df

def de_duplication(df):
    # Remove duplicate rows
    df = df.drop_duplicates()
    df = df.reset_index(drop=True)
    return df

def noise_remover(df):
    # Keep only rows where all three target columns have valid values
    df = df[df[Config.TYPE_2_COL].notna()]
    df = df[df[Config.TYPE_3_COL].notna()]
    df = df[df[Config.TYPE_4_COL].notna()]

    # Remove classes with too few instances
    for col in [Config.TYPE_2_COL, Config.TYPE_3_COL, Config.TYPE_4_COL]:
        counts = df[col].value_counts()
        valid = counts[counts >= Config.MIN_CLASS_COUNT].index
        df = df[df[col].isin(valid)]

    df = df.reset_index(drop=True)
    return df

def translate_to_en(texts):
    # Placeholder — returns texts as-is (translation not required for this CA)
    return texts
    