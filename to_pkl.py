import pyreadr
import pandas as pd

def load_rds_to_pickle(rds_path, pkl_path):
    """Load an RDS file and save it as a pickle file."""
    result = pyreadr.read_r(rds_path) 
    df = result[None]  # Extract the pandas DataFrame
    df.to_pickle(pkl_path)  # Save DataFrame as a pickle file
    print(f"Saved {rds_path} to {pkl_path}")

    return df

def main():
    # Data importation and saving
    daily = load_rds_to_pickle('wrds_data/crsp_daily_full.rds', "pkl/daily.pkl")
    df = load_rds_to_pickle('wrds_data/compustat_all.rds', "pkl/compu_all.pkl")

if __name__ == "__main__":
    main()