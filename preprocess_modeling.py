from process import processing_na

import pandas as pd
from datetime import datetime, timedelta

def load_and_clean_data():
    '''
    This function loads directly the processed dataset (all preprocessing is done in process.py).  
    '''
    dataset = (
        pd.read_csv('processed_set.csv', sep = ',', index_col = 0)
        .assign(
            DateFiled = lambda df : pd.to_datetime(df.DateFiled),
            publication_date = lambda df : pd.to_datetime(df.publication_date),
            publication_date_end = lambda df : pd.to_datetime(df.publication_date_end)
        )  
    )
    
    dataset = processing_na(dataset)
    return dataset


def clean_dataset(df, variables_to_keep: list) -> pd.DataFrame:
    ''' dropna sauf sur la colonne DateFiled. C'est brutforce, voir comment faire plus proprement '''
    return df.filter(variables_to_keep).dropna(subset=[col for col in variables_to_keep if col != "DateFiled"])

def preprocess_training_test(df: pd.DataFrame, split_date: str, training_time: int = 5, live: bool = False) -> (pd.DataFrame, pd.DataFrame):
    '''
    Filters the dataframe to create training and testing datasets where 'publication_date_end'
    is between (split_date - training_time years) and split_date for training,
    and after split_date for testing.
    
    Parameters:
    df (pd.DataFrame): Input dataframe 
    split_date (str): The cutoff date in "YYYY-MM-DD" format.
    training_time (int): Number of years before the split_date to include in training.
    live (bool): If True, we remove in df_test the bankruptcy dates before split_date.
    
    Returns:
    Tuple[pd.DataFrame, pd.DataFrame]: Filtered training and testing dataframes.
    '''
    
    split_dt = datetime.strptime(split_date, "%Y-%m-%d")
    start_date = split_dt.replace(year=split_dt.year - training_time)
    df["publication_date_end"] = pd.to_datetime(df["publication_date_end"], errors='coerce')
    
    df_train = df[(df["publication_date_end"] >= start_date) & (df["publication_date_end"] <= split_dt)]
    
    if live:
        # we drop rows where bankruptcy date is before split_date (in live, we don't need to predict the past)
        df_test = df[(df["publication_date"] <= split_dt) & (df["publication_date_end"] > split_dt) & ((df["DateFiled"] >= split_dt) | (df["DateFiled"].isnull()))]
    else:
        df_test = df[(df["publication_date"] <= split_dt) & (df["publication_date_end"] > split_dt)]
        
    return df_train, df_test

def merge_market_data(df_train, df_test):
    '''
    Tom régale toi
    
    Pense à enlever DateFiled, publication_date, publication_date_end et conm à la fin sur df_train, df_test stp !!!
    '''
    pass

    return df_train, df_test

def main():
    df = load_and_clean_data()
    
    # we define predictors as we want
    variables_to_keep = ['conm','DateFiled','publication_date','publication_date_end','WCTA','RETA','EBTA','TLTA','SLTA','default']
    # we define split_date as we want (end of year is preferable)
    split_date = "2020-12-31"
    
    df_cleaned = clean_dataset(df, variables_to_keep)
    df_train, df_test = preprocess_training_test(df_cleaned, split_date, live=False)
    
if __name__ == "__main__":
    main()
