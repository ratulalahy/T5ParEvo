from sklearn.model_selection import train_test_split
from pandas import DataFrame
from typing import Tuple

class DatasetPreparation:
    def __init__(self, df: DataFrame, split_size: float):
        self.df = df
        self.split_size = split_size

    def split_and_reset_index(self) -> Tuple[DataFrame, DataFrame]:
        """Splits the dataframe into train and validation sets and resets their indices."""
        df_train, df_val = train_test_split(self.df, test_size=self.split_size)
        df_train.reset_index(drop=True, inplace=True)
        df_val.reset_index(drop=True, inplace=True)
        return df_train, df_val