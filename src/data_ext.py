import os
# import boto3
import pandas as pd
from src import config
from typing import Tuple
from sklearn.model_selection import train_test_split

# s3 = boto3.resource('s3', aws_access_key_id=config.access_key, aws_secret_access_key=config.secret_key)

# def get_dataset() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
#     """
#     Download from S3 all the needed datasets for the project.

#         variable_list : pd.DataFrame
#             Extra dataframe with detailed description about dataset features
#     """
#     if not os.path.exists("./dataset"):
#         os.makedirs("./dataset")

#     for obj in s3.Bucket(config.bucket_name).objects.filter(Prefix=config.key):
#         if not obj.key.endswith('/'):
#             s3.meta.client.download_file(config.bucket_name, 
#                                         obj.key, 
#                                         f'dataset/{obj.key.split("/")[-1]}')
    
#     #Variable List
#     variable_list = pd.read_excel("dataset/PAKDD2010_VariablesList.XLS")
#     variable_list.loc[43,'Var_Title']='MATE_EDUCATION_LEVEL'
#     columns_names = variable_list['Var_Title'].tolist()

#     #Modeling Data
#     train_df = pd.read_csv('dataset/PAKDD2010_Modeling_Data.txt', 
#                               encoding='latin-1',
#                               delimiter="\t",
#                               low_memory=False,
#                               header=None,
#                               names=columns_names)
    
#     #Predicition data
#     test_df = pd.read_csv('dataset/PAKDD2010_Prediction_Data.txt', 
#                               encoding='latin-1',
#                               delimiter="\t",
#                               header=None,
#                               names=columns_names)
    
#     return train_df, test_df, variable_list

def get_features(train_df: pd.DataFrame, test_df: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Separates our train and test datasets columns between Features
    (the input to the model) and Targets (what the model has to predict with the
    given features).

    Arguments:
        train_df : pd.DataFrame
            Training dataset
        test_df : pd.DataFrame
            Testing dataset
    Returns:
        X_train : pd.DataFrame
            Training features
        y_train : pd.Series
            Training target
        X_test : pd.DataFrame
            Testing features
        y_test : pd.Series
            Testing target
    """
    X_train = train_df.drop('TARGET_LABEL_BAD=1', axis=1)
    y_train = train_df['TARGET_LABEL_BAD=1']
    X_test = test_df.drop('TARGET_LABEL_BAD=1', axis=1)
    y_test = test_df['TARGET_LABEL_BAD=1']

    return X_train, y_train, X_test, y_test

def get_train_val(X_train: pd.DataFrame, y_train: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Split training dataset in two new sets used for train and validation.

    Arguments:
        X_train : pd.DataFrame
            Original training features
        y_train: pd.Series
            Original training labels/target

    Returns:
        X_train : pd.DataFrame
            Training features
        X_val : pd.DataFrame
            Validation features
        y_train : pd.Series
            Training target
        y_val : pd.Series
            Validation target
    """ 
    X_train, X_val, y_train, y_val = train_test_split(X_train,
                                                      y_train,
                                                      test_size=0.2,
                                                      random_state=42,
                                                      shuffle=True)
     
    return X_train, X_val, y_train, y_val