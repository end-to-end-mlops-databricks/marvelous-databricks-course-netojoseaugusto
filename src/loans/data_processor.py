import pandas as pd
from typing import List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class DataBuilder:
    def __init__(self, dataframe: Optional[pd.DataFrame] = None):
        """
        Initializes the DataBuilder with an optional DataFrame.
        """
        self.dataframe: Optional[pd.DataFrame] = dataframe
        self.features: Optional[pd.DataFrame] = None
        self.target: Optional[pd.Series] = None

    def load_data(self, path: str) -> 'DataBuilder':
        """
        Loads a CSV file into the dataframe attribute.
        """
        self.dataframe = pd.read_csv(path)
        return self

    def drop_columns(self, column_names: List[str]) -> 'DataBuilder':
        """
        Drops specified columns from the dataframe attribute.
        """
        if self.dataframe is None:
            raise ValueError("Dataframe is not loaded. Call load_data() before dropping columns.")
        self.dataframe.drop(columns=column_names, inplace=True)
        return self

    def separate_features_and_target(self, target_column: str) -> 'DataBuilder':
        """
        Separates the dataframe into features and target attributes.
        """
        if self.dataframe is None:
            raise ValueError("Dataframe is not loaded. Call load_data() before separating features and target.")
        if target_column not in self.dataframe.columns:
            raise ValueError(f"Target column '{target_column}' does not exist in the dataframe.")
        self.features = self.dataframe.drop(columns=[target_column])
        self.target = self.dataframe[target_column]
        return self

    def get_dataframe(self) -> pd.DataFrame:
        """
        Returns the current dataframe.
        """
        if self.dataframe is None:
            raise ValueError("Dataframe is not loaded.")
        return self.dataframe

    def get_features_and_target(self) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Returns the features and target as a tuple.
        """
        if self.features is None or self.target is None:
            raise ValueError("Features and target have not been separated. Call separate_features_and_target() first.")
        return self.features, self.target