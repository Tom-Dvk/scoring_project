from preprocess_modeling import load_and_clean_data, preprocess_training_test, clean_dataset

import pandas as pd
import numpy as np

from sklearn.linear_model import Lasso
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

def get_variables_lasso(df:pd.DataFrame, split_date:str, training_time: int = 5, live: bool = False, full_sample: bool = True, finetuning: bool = False, alpha: float=0.1)-> list:
    '''
    args:
    - df (pd.DataFrame): Input dataframe
    - split_date (str): The cutoff date in "YYYY-MM-DD" format.
    - training_time (int): Number of years before the split_date to include in training.
    - live (bool): If True, we remove in df_test the bankruptcy dates before split_date.
    - full_sample (bool): If True, we use the full sample to train the model.
    - finetuning (bool): It True, estimate alpha with gridsearch
    - alpha (float): Regularization parameter for Lasso.
    
    Returns:
    List[str]: List of selected variables
    '''
    variables_to_keep = [col for col in df.columns if col not in ['conm','GvkeyBefore']]
    ratios = ['WCTA','RETA','EBTA','TLTA','SLTA','r1','r2','r3','r4','r5','r6','r7','r8','r9','r10','r11','r12','r13','r14','r15','r16','r17','r18','r19','r20','WCTA_lag','RETA_lag','EBTA_lag','TLTA_lag','SLTA_lag','r1_lag','r2_lag','r3_lag','r4_lag','r5_lag','r6_lag','r7_lag','r8_lag','r9_lag','r10_lag','r11_lag','r12_lag','r13_lag','r14_lag','r15_lag','r16_lag','r17_lag','r18_lag','r19_lag','r20_lag']
    
    df[ratios] = df.sort_values(by=['conm', 'publication_date']).groupby('conm')[ratios].ffill()
    df[ratios] = df.sort_values(by=['conm', 'publication_date']).groupby('conm')[ratios].bfill()

    df_cleaned = clean_dataset(df, variables_to_keep)
    
    if full_sample:
        df_train = df_cleaned
    else:
        df_train = preprocess_training_test(df_cleaned, split_date, training_time=training_time, live=live)[0]
    
    X = df_train.drop("default", axis=1).drop(['publication_date','publication_date_end','DateFiled'], axis=1)
    y = df_train["default"]
    
    if finetuning:
        best_params, selected_features = fine_tuning(X, y, Lasso())
    else:
        model = Lasso(alpha=alpha)
        model.fit(X, y)
        
        selected_features = X.columns[model.coef_ != 0].tolist()
    

    return selected_features

def fine_tuning(X, y, model, cv=5, scoring='neg_mean_squared_error'):
    """
    Fine-tunes a given model using GridSearchCV and returns selected features.
    
    Parameters:
        X (array-like): Feature matrix.
        y (array-like): Target vector.
        model (estimator): Model to fine-tune.
        cv (int): Number of cross-validation folds.
        scoring (str): Scoring metric.
    
    Returns:
        dict: Best hyperparameters found.
        list: Selected feature names (non-zero coefficients).
    """
    
    param_grid = {'model__alpha': [0.001, 0.01, 0.1, 1, 10, 100]}
    feature_names = X.columns.tolist() 
    
    # Create pipeline (scaling + model)
    pipeline = Pipeline([
        ('scaler', StandardScaler()), 
        ('model', model)
    ])
    
    grid_search = GridSearchCV(pipeline, param_grid, cv=cv, scoring=scoring)
    grid_search.fit(X, y)
    
    best_model = grid_search.best_estimator_.named_steps['model']
    
    selected_indices = np.where(best_model.coef_ != 0)[0]
    
    if selected_indices.size == 0:
        print(f"No features selected with alpha={best_model.alpha}")
        
    selected_features = [feature_names[i] for i in selected_indices]
    
    return grid_search.best_params_, selected_features

    

def main():
    df = load_and_clean_data()
    
    # if full_sample is True, we use the full sample for selection variable (i.e. split_date, training_time and live are not used)
    # if finetuning is True, we estimate alpha with gridsearch (i.e. alpha is not used)
    selected_variables = get_variables_lasso(df, split_date="2020-12-31", training_time=3, live=False, full_sample=True, finetuning=True, alpha=0.1)
    print(selected_variables)
if __name__ == "__main__":
    main()

