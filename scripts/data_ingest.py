# Convert CSV to Parquet for faster I/O.

import os
import pandas as pd

def load_raw(path: str) -> pd.DataFrame:
    """
    Load the raw CSV Superstore dataset and return as DataFrame.
    low_memory=False prevents mixed-type warnings.
    """
    df = pd.read_csv(path, low_memory=False)
    return df


if __name__ == '__main__':
    # Hardcoded paths (your directory)
    infile = r"D:\Data analysis BI\Project\Superstore_analysis\data\raw\Sample - Superstore.csv"
    outfile = r"D:\Data analysis BI\Project\Superstore_analysis\data\raw\superstore.parquet"

    # Load CSV
    df = load_raw(infile)

    # Ensure output directory exists
    os.makedirs(os.path.dirname(outfile), exist_ok=True)

    # Save to Parquet for faster reads later
    df.to_parquet(outfile)
    print(f"Saved ingested data to {outfile}")
