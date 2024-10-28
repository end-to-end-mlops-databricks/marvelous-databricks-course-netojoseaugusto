import pandas as pd
from typing import List

class DataBuilder:
    def __init__(self):
        self.dataframe = None
        self.features = None
        self.target = None

    def load_data(self, path: str):
        """
        Load a CSV file into the dataframe attribute.
        """
        self.dataframe = pd.read_csv(path)
        return self

    def drop_columns(self, column_names: List[str]):
        """
        Drops specified columns from the dataframe attribute.
        """
        if self.dataframe is not None:
            self.dataframe = self.dataframe.drop(columns=column_names)
        else:
            raise ValueError("Dataframe is not loaded. Call load_data() before dropping columns.")
        return self

    def separate_features_and_target(self, target_column: str):
        """
        Separates the dataframe into features and target attributes.
        """
        if self.dataframe is not None:
            self.features = self.dataframe.drop(columns=[target_column])
            self.target = self.dataframe[[target_column]]
        else:
            raise ValueError("Dataframe is not loaded. Call load_data() before separating features and target.")
        return self

    def get_dataframe(self) -> pd.DataFrame:
        return self.dataframe

    def get_features_and_target(self) -> (pd.DataFrame, pd.DataFrame):
        return self.features, self.target
