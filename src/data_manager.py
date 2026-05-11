from pathlib import Path
from pandas import DataFrame, Series
import pandas as pd


class DataManager:
    def __init__(
        self,
        genomic_file_path: Path,
        metadata_file_path: Path,
        target_column: str,
        metadata_file_index: str,
    ):
        self.genomic_file_path: Path | str = genomic_file_path
        self.metadata_file_path: Path | str = metadata_file_path
        self.merged_df: DataFrame | None = None
        self.target_column: str = target_column
        self.metadata_file_index = metadata_file_index

    def merge_datasets(self) -> None:
        """
        Loads and merges datasets, then stores it in merged_df attribute.
        Expects genomic file to have samples as columns, and metadata file
        to have samples as rows.

        WARNING/TO-DO:
        Currently, this method deals with perks particular to my data. I need
        to think/discuss how this will be handled once we deploy this as an app.
        Perks: index_col
        """

        genomic_df = pd.read_csv(self.genomic_file_path, index_col=0).T
        metadata_df = pd.read_csv(
            self.metadata_file_path, index_col=self.metadata_file_index
        )[self.target_column]

        merged_df = pd.merge(
            genomic_df, metadata_df, how="inner", right_index=True, left_index=True
        )

        self.merged_df = merged_df

    def get_X_y(self) -> tuple[DataFrame, Series]:
        if self.merged_df is None:
            raise RuntimeError(
                "Data has not been loaded yet. Use load_data() before get_X_y"
            )

        merged_df = self.merged_df
        target_column = self.target_column
        X = merged_df.drop(target_column, axis="columns")
        y = merged_df[target_column]

        if type(y) is not Series:
            raise RuntimeError(
                f"Target column is of type {type(y)}, not Pandas.Series."
            )

        return (X, y)
