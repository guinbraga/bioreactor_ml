from data_manager import DataManager

GENOMIC_DATA_PATH = "../data/map_complete_absolute_n_hits_table.csv"
METADATA_PATH = "../data/metadata_enrichment.csv"

data_manager = DataManager(
    genomic_file_path=GENOMIC_DATA_PATH,
    metadata_file_path=METADATA_PATH,
    target_column="EXPERIMENT",
    genomic_file_index="Cluster sample ID",
)
