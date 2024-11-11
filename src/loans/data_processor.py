import logging
from typing import List, Optional, Tuple

import pandas as pd

from pyspark.sql import SparkSession
from pyspark.sql.functions import current_timestamp, to_utc_timestamp

logger = logging.getLogger(__name__)


class DataBuilder:
    def __init__(self, dataframe: Optional[pd.DataFrame] = None):
        """
        Initializes the DataBuilder with an optional DataFrame.
        """
        self.dataframe: Optional[pd.DataFrame] = dataframe
        self.features: Optional[pd.DataFrame] = None
        self.target: Optional[pd.Series] = None

    def load_data(self, path: str) -> "DataBuilder":
        """
        Loads a CSV file into the dataframe attribute.
        """
        self.dataframe = pd.read_csv(path)
        return self

    def drop_columns(self, column_names: List[str]) -> "DataBuilder":
        """
        Drops specified columns from the dataframe attribute.
        """
        if self.dataframe is None:
            raise ValueError("Dataframe is not loaded. Call load_data() before dropping columns.")
        self.dataframe.drop(columns=column_names, inplace=True)
        return self

    def separate_features_and_target(self, target_column: str) -> "DataBuilder":
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

    def save_dataset(self, unity_catalog_location: str, spark: SparkSession) -> None:
        """
        Saves the current dataframe to a Unity Catalog location using Delta Lake.
        
        Parameters:
        - unity_catalog_location (str): The target location in Unity Catalog to save the dataset.
        - spark (SparkSession): The Spark session to use for writing the data.

        Raises:
        - ValueError: If the dataframe is not loaded or invalid Unity Catalog location is provided.
        """
        if self.dataframe is None:
            raise ValueError("Dataframe is not loaded. Please load the data before saving.")
        if not unity_catalog_location:
            raise ValueError("Unity Catalog location must be provided.")
        
        try:
            # Convert the Pandas DataFrame to a Spark DataFrame and add a timestamp column
            spark_dataframe = spark.createDataFrame(self.dataframe).withColumn(
                "update_timestamp_utc", to_utc_timestamp(current_timestamp(), "UTC")
            )
            
            # Write the data to the specified Unity Catalog location
            spark_dataframe.write.mode("append").format("delta").saveAsTable(unity_catalog_location)
            
            # Enable change data feed on the target table
            spark.sql(
                f"ALTER TABLE {unity_catalog_location} "
                "SET TBLPROPERTIES (delta.enableChangeDataFeed = true)"
            )
        except Exception as e:
            raise RuntimeError(f"Failed to save the dataset to Unity Catalog: {e}")
