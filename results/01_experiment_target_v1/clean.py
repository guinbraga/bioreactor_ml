import pandas as pd
from pathlib import Path


def clean_brackets():
    # Get the directory where the script is currently running
    current_dir = Path.cwd()

    # Create a new directory for the fixed files to protect your originals
    output_dir = current_dir / "cleaned_csvs"
    output_dir.mkdir(exist_ok=True)

    # Find all CSV files in the folder
    csv_files = list(current_dir.glob("*.csv"))

    if not csv_files:
        print("No CSV files found in this directory.")
        return

    print(f"Found {len(csv_files)} CSV files. Starting cleanup...\n")

    for file_path in csv_files:
        print(f" -> Fixing: {file_path.name}")

        # Load the CSV
        df = pd.read_csv(file_path)

        # Identify columns that contain text/strings (Pandas calls them 'object')

        # .str.strip() removes specific characters ONLY from the very edges of the string.
        # This perfectly removes `[`, `]`, `'`, and `"` without hurting the inner text.
        df["Predicted Class"] = df["Predicted Class"].str.strip("[]'\"")

        # Save the fixed dataframe to the new folder (index=False prevents adding a duplicate ID column)
        output_path = output_dir / file_path.name
        df.to_csv(output_path, index=False)

    print(
        f"\n✅ All done! Your clean files are waiting in the '{output_dir.name}' folder."
    )


if __name__ == "__main__":
    clean_brackets()
