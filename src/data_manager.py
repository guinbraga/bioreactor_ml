import itertools
from pathlib import Path
from pandas import DataFrame, Series
import pandas as pd


class DataManager:
    def __init__(
        self,
        genomic_file_path: Path | str,
        metadata_file_path: Path | str,
        metadata_file_index: str,
        target_columns: list[str],
    ):
        self.genomic_file_path: Path | str = genomic_file_path
        self.metadata_file_path: Path | str = metadata_file_path
        self.merged_df: DataFrame | None = None
        self.target_columns: list[str] = target_columns
        self.metadata_file_index: str = metadata_file_index

    @staticmethod
    def get_metadata_columns(file_path: Path | str) -> list:
        metadata_df = pd.read_csv(file_path, nrows=0)
        columns = metadata_df.columns.tolist()
        return columns

    def merge_datasets(self) -> dict:
        """
        Loads and merges datasets, then stores it in merged_df attribute.
        Expects genomic file to have samples as columns, and metadata file
        to have samples as rows.

        Returns a dictionary payload to be passed to the front-end, as a way
        of presenting any an overall summary of the merge process to the user.
        """

        genomic_df = pd.read_csv(self.genomic_file_path, index_col=0).T
        metadata_df = pd.read_csv(
            self.metadata_file_path, index_col=self.metadata_file_index
        )[self.target_columns]

        merged_df = pd.merge(
            genomic_df, metadata_df, how="inner", right_index=True, left_index=True
        )

        self.merged_df = merged_df

        original_samples = set(itertools.chain(genomic_df.index, metadata_df.index))
        merged_samples = merged_df.index
        excluded_samples = [
            sample for sample in original_samples if sample not in merged_samples
        ]

        merge_results = {
            "genomic_n_samples": len(genomic_df),
            "metadata_n_samples": len(metadata_df),
            "merged_n_samples": len(merged_df),
            "excluded_samples": excluded_samples,
        }

        return merge_results

    def get_X_y(self, target_column) -> tuple[DataFrame, Series]:
        if self.merged_df is None:
            raise RuntimeError(
                "Data has not been loaded yet. Use load_data() before get_X_y"
            )

        merged_df = self.merged_df
        X = merged_df.drop(self.target_columns, axis="columns")
        y = merged_df[target_column]

        if type(y) is not Series:
            raise RuntimeError(
                f"Target column is of type {type(y)}, not Pandas.Series."
            )

        return (X, y)

    def get_groups(self, group_column: str | None) -> Series | None:
        if group_column is None:
            return None
        metadata_df = pd.read_csv(self.metadata_file_path)
        groups = metadata_df[group_column]
        if type(groups) is not Series:
            raise TypeError(
                "Groups is not a Series. Either group_column is not a metadata column or is a list of columns."
            )
        return groups
