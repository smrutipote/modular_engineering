import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from Config import Config
import random

seed = 0
random.seed(seed)
np.random.seed(seed)

class Data():
    def __init__(self, X: np.ndarray, df: pd.DataFrame) -> None:
        # Split into train and test
        self.X_train, self.X_test, self.train_df, self.test_df = train_test_split(
            X, df, test_size=0.2, random_state=seed
        )
        self.embeddings = X

        # Set y based on the target column (Type 2 by default)
        self.y = df[Config.TYPE_2_COL]
        self.y_train = self.train_df[Config.TYPE_2_COL]
        self.y_test = self.test_df[Config.TYPE_2_COL]

    def get_type(self):
        return self.y

    def get_X_train(self):
        return self.X_train

    def get_X_test(self):
        return self.X_test

    def get_type_y_train(self):
        return self.y_train

    def get_type_y_test(self):
        return self.y_test

    def get_train_df(self):
        return self.train_df

    def get_embeddings(self):
        return self.embeddings

    def get_type_test_df(self):
        return self.test_df