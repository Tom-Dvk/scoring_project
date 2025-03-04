import pandas as pd
import pyreadr
import numpy as np



#### Defining paths 

compustat_path = "../wrds_data/compustat_all.rds"
crsp_daily_path = "../pkl/daily.pkl"
company_path = "../wrds_data/company_all.rds"
linktable_path = "../wrds_data/ccmxpf_linktable.rds"
lopucki_path = "../default_data/Bankruptcy - LoPucki/Florida-UCLA-LoPucki Bankruptcy Research Database 1-12-2023.xlsx"

####

def load_compustat_data(filepath):
    """Load and process Compustat data."""
    df = pyreadr.read_r(filepath)[None]
    df = df.assign(
        datadate=lambda x: pd.to_datetime(x.datadate, format="%Y-%m-%d"),
        fdate = lambda x: pd.to_datetime(x.fdate, format="%Y-%m-%d"),
        WCTA = lambda df: df["wcap"] / df["at"],
        RETA = lambda df: df["re"] / df["at"],
        EBTA = lambda df: df["ebit"] / df["at"],
        TLTA = lambda df: df["lt"] / df["at"],
        SLTA = lambda df: df["sale"] / df["at"],
        r1=lambda df: (df['dltt'] + df['dlc']) / df['at'],
        r2=lambda df: df['seq'] / (df['dltt'] + df['dlc'] + df['seq'] - df['che']),
        r3=lambda df: df['dlc'] / (df['dlc'] + df['dltt']),
        r4=lambda df: df['dlc'] / df['at'],
        r5=lambda df: df['dltt'] / df['at'],
        r6=lambda df: (df['dltt'] + df['dlc']) / df['revt'],
        r7=lambda df: df['dltt'] / df['revt'],
        r8=lambda df: df['dlc'] / df['revt'],
        r9=lambda df: (df['act'] - df['lct']) / df['revt'],
        r10=lambda df: df['dlc'] / df['act'],
        r11=lambda df: df['xint'] / (df['dltt'] + df['dlc']),
        r12=lambda df: df['xint'] / df['revt'],
        r13=lambda df: df['ebitda'] / df['at'],
        r14=lambda df: df['ebitda'] / df['revt'],
        r15=lambda df: df['ppent'] / df['revt'],
        r16=lambda df: (df['xint'] + df['dltr']) / df['revt'],
        r17=lambda df: df['xint'] / df['ebitda'],
        r18=lambda df: df['ni'] / df['revt'],
        r19=lambda df: (df['act'] - df['lct']) / ((df['revt'] - df['gp']) - df['xint']),
        r20=lambda df: df['revt'] / df['at']
    )
    return df

def load_crsp_daily(filepath):
    """Load and process CRSP daily data."""
    df = pd.read_pickle(filepath)
    df = df.assign(date=lambda x: pd.to_datetime(x.date, format="%Y-%m-%d"))
    return df

def load_dataset(filepath):
    """Generic function to load RDS datasets."""
    return pyreadr.read_r(filepath)[None]

def load_lopucki_data(filepath):
    """Load and process LoPucki bankruptcy data."""
    df = pd.read_excel(filepath)
    df = df.assign(
        GvkeyBefore=lambda x: x['GvkeyBefore'].astype(str),
        DateFiled=lambda x: pd.to_datetime(x['DateFiled'])
    ).filter(['NameCorp', 'Chapter', 'GvkeyBefore', 'DateFiled'])
    return df

def merge_in_chunks1(df1, df2, on, how='left', chunk_size=100000):
    """Merge two large DataFrames in chunks to prevent memory issues."""
    merged_chunks = []
    for i in range(0, len(df1), chunk_size):
        chunk = df1.iloc[i:i+chunk_size]
        merged_chunk = chunk.merge(df2, on=on, how=how, indicator=True)
        merged_chunks.append(merged_chunk)
    return pd.concat(merged_chunks, ignore_index=True)

def merge_in_chunks2(df1, df2, chunk_size=100000):
    """Merge two large DataFrames in chunks to prevent memory issues."""
    merged_chunks = []
    for i in range(0, len(df1), chunk_size):
        chunk = df1.iloc[i:i+chunk_size]
        merged_chunk = chunk.merge(df2, left_on='gvkey', right_on='GvkeyBefore', how='left')
        merged_chunks.append(merged_chunk)
    return pd.concat(merged_chunks, ignore_index=True)

def processing_na(df):
    df.replace([float('inf'), float('-inf')], float('nan'), inplace=True)
    df = (
        df
        .groupby("conm")
        .ffill()
    ).join(df[['conm']])
    return df

def process_data(acc, crsp_daily, ccmxpf_linktable, company_variable, lopucki):
    """Process datasets after loading."""
    print('STEP 1')
    acc = (
        acc
        .filter(['gvkey','fyear','datadate','fdate','conm','WCTA','RETA','EBTA','TLTA','SLTA', 'r1', 'r2', 'r3', 'r4','r5','r6','r7','r8','r9','r10','r11','r12','r13','r14','r15','r16','r17','r18','r19','r20'])
        .assign(fyear = lambda df : pd.to_datetime(df['fyear'], format='%Y').dt.year)
        )
    # Mapping subset
    subset_mapping = (
        ccmxpf_linktable
        .query(f'gvkey in {acc.gvkey.unique().tolist()}')
        .merge(
            company_variable.filter(['gvkey', 'conm']),
            how='left',
            on='gvkey'
        )
    )
    print('STEP 2')
    # Selecting specific columns from crsp_daily
    crsp_daily_selected = crsp_daily[['cusip', 'permno', 'date', 'prc', 'vol', 'shrout', 'bid', 'ask']]
    
    # Merging in chunks with subset_mapping using 'permno'
    merged_df = merge_in_chunks1(crsp_daily_selected, subset_mapping, on='permno', how='left')

    # Filtering rows where 'gvkey' is not null
    merged_crsp_compustat_sub = merged_df[merged_df['gvkey'].notna()]
    print('STEP 3')
    # Processing LoPucki bankruptcy data
    lopucki_clean = (
        lopucki
        .query("Chapter in ['7', '11']")
        .assign(
            DateFiled=lambda x: pd.to_datetime(x.DateFiled, format='%m/%d/%Y'),
            GvkeyBefore=lambda x: x.GvkeyBefore.astype(int)
        )
        .sort_values(by=['GvkeyBefore', 'DateFiled'])
        .groupby('GvkeyBefore', as_index=False)
        .agg(
            DateFiled=('DateFiled', 'min'),
            NameCorp=('NameCorp', 'first'),
            Chapter=('Chapter', 'first')
        )
    )
    print('STEP 4')
    # Modifying acc dataset
    acc_modified = (
        acc
        .assign(
            publication_date=lambda df: np.where(df.fdate.isna(), df.datadate + pd.DateOffset(months=4), df.fdate),
            publication_date_end=lambda df: df.publication_date + pd.DateOffset(months=12),
            gvkey=lambda df: df.gvkey.astype(int)
        )
        .drop(['fyear', 'datadate', 'fdate'], axis=1)
    )
    print('STEP 5')
    # Merging Compustat with LoPucki bankruptcy data in chunks
    dataset = merge_in_chunks2(acc_modified, lopucki_clean[['GvkeyBefore', 'DateFiled']])
    
    # Assigning default status
    dataset = dataset.assign(
       default = lambda df : np.where((df.DateFiled >= df.publication_date) & (df.DateFiled < df.publication_date_end),1,0)).sort_values(by=['gvkey','publication_date']).assign( 
        WCTA_lag = lambda df: df.groupby('gvkey')['WCTA'].shift(1),
        RETA_lag = lambda df: df.groupby('gvkey')['RETA'].shift(1),
        EBTA_lag = lambda df: df.groupby('gvkey')['EBTA'].shift(1),
        TLTA_lag = lambda df: df.groupby('gvkey')['TLTA'].shift(1),
        SLTA_lag = lambda df: df.groupby('gvkey')['SLTA'].shift(1),
        r1_lag = lambda df: df.groupby('gvkey')['r1'].shift(1),
        r2_lag = lambda df: df.groupby('gvkey')['r2'].shift(1),
        r3_lag = lambda df: df.groupby('gvkey')['r3'].shift(1),
        r4_lag = lambda df: df.groupby('gvkey')['r4'].shift(1),
        r5_lag = lambda df: df.groupby('gvkey')['r5'].shift(1),
        r6_lag = lambda df: df.groupby('gvkey')['r6'].shift(1),
        r7_lag = lambda df: df.groupby('gvkey')['r7'].shift(1),
        r8_lag = lambda df: df.groupby('gvkey')['r8'].shift(1),
        r9_lag = lambda df: df.groupby('gvkey')['r9'].shift(1),
        r10_lag = lambda df: df.groupby('gvkey')['r10'].shift(1),
        r11_lag = lambda df: df.groupby('gvkey')['r11'].shift(1),
        r12_lag = lambda df: df.groupby('gvkey')['r12'].shift(1),
        r13_lag = lambda df: df.groupby('gvkey')['r13'].shift(1),
        r14_lag = lambda df: df.groupby('gvkey')['r14'].shift(1),
        r15_lag = lambda df: df.groupby('gvkey')['r15'].shift(1),
        r16_lag = lambda df: df.groupby('gvkey')['r16'].shift(1),
        r17_lag = lambda df: df.groupby('gvkey')['r17'].shift(1),
        r18_lag = lambda df: df.groupby('gvkey')['r18'].shift(1),
        r19_lag = lambda df: df.groupby('gvkey')['r19'].shift(1),
        r20_lag = lambda df: df.groupby('gvkey')['r20'].shift(1)
        ).groupby('gvkey').ffill()
    
    return dataset, merged_crsp_compustat_sub, lopucki_clean

def main():

    company_all = load_compustat_data(compustat_path)
    crsp_daily = load_crsp_daily(crsp_daily_path)
    company_variable = load_dataset(company_path)
    ccmxpf_linktable = load_dataset(linktable_path)
    lopucki = load_lopucki_data(lopucki_path)
    dataset ,merged_crsp_compustat_sub, lopucki_clean = process_data(company_all, crsp_daily, ccmxpf_linktable, company_variable, lopucki)
    
    dataset = processing_na(dataset)


    dataset.to_csv('processed_set.csv')

if __name__ == "__main__":
    main()

####
#### CODE EN DESSOUS EST TROP LONG, PERMETRRAIT DE FAIRE JOINTURE DONNEE DE MARCHE ET DONNEE COMPTABLE ####
#### 

# def calculate_percentage_change(first_value, last_value):
#     """Calcule la variation en pourcentage entre deux valeurs."""
#     if first_value is None or last_value is None or first_value == 0:
#         return None
#     return ((last_value - first_value) / abs(first_value)) * 100

# def get_market_data(market_df, gvkey, date):
#     """Retourne la variation en pourcentage et la volatilité du prix sur un an pour un gvkey donné."""
#     date = pd.to_datetime(date)
#     start_date = date - pd.DateOffset(years=1)

#     # Filtrer les données pertinentes
#     filtered_market_df = market_df.loc[
#         (market_df['gvkey'] == gvkey) & 
#         (market_df['date'] >= start_date) & 
#         (market_df['date'] < date)
#     ]

#     if filtered_market_df.empty:
#         return None, None

#     # Trier par date pour garantir la bonne sélection des valeurs
#     filtered_market_df = filtered_market_df.sort_values(by='date')

#     # Récupérer la première et la dernière valeur de `prc`
#     first_prc = filtered_market_df['prc'].iloc[0]
#     last_prc = filtered_market_df['prc'].iloc[-1]

#     # Calculer les statistiques
#     percentage_change = calculate_percentage_change(first_prc, last_prc)
#     coef_variation = filtered_market_df['prc'].var()  / filtered_market_df['prc'].mean()

#     return percentage_change, coef_variation

# # Appliquer la fonction et stocker les résultats dans deux nouvelles colonnes
# dataset[['percentage_change', 'volatility']] = dataset.apply(
#     lambda row: pd.Series(get_market_data(merged_crsp_compustat_sub, row['GvkeyBefore'], row['publication_date'])), axis=1)