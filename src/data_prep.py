import numpy as np
import pandas as pd

from typing import Tuple
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, OrdinalEncoder

def preprocess_data(
    train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

    #Copy of dataframes
    working_train_df = train_df.copy()
    working_val_df = val_df.copy()
    working_test_df = test_df.copy()

    #Remove and modify unnecesary columns
    cols_to_drop=['ID_CLIENT',
                        'MATE_EDUCATION_LEVEL',
                        'MATE_PROFESSION_CODE',
                        'CLERK_TYPE',
                        'FLAG_MOBILE_PHONE',
                        'FLAG_ACSP_RECORD']
    working_train_df['SEX'] = working_train_df['SEX'].replace(' ', 'F')
    working_train_df['APPLICATION_SUBMISSION_TYPE'] = working_train_df['APPLICATION_SUBMISSION_TYPE'].replace('0', 'Web')
    working_train_df=working_train_df.drop(cols_to_drop,axis=1)
    working_val_df=working_val_df.drop(cols_to_drop,axis=1)
    working_test_df=working_test_df.drop(cols_to_drop,axis=1)

    #Binary and Multi-Class Columns
    binary_cols=[]
    multi_cols=[]
    categorical_cols = working_train_df.select_dtypes(include='object').columns.tolist()
    for cat_col in categorical_cols:
            unique_values = working_train_df[cat_col].nunique()
            if unique_values==1 or unique_values>=20:
                working_train_df=working_train_df.drop(cat_col,axis=1)
                working_val_df=working_val_df.drop(cat_col,axis=1)
                working_test_df=working_test_df.drop(cat_col,axis=1)
            if unique_values==2:
                binary_cols.append(cat_col)
            if unique_values>=3 and unique_values<20:
                multi_cols.append(cat_col)

    #OrdinalEncoder for Binary Columns
    input_df=[working_train_df,working_val_df,working_test_df]
    ord_encoder = OrdinalEncoder()
    for col in binary_cols:
        for df in input_df:
            df[col] = ord_encoder.fit_transform(df[[col]].fillna('Unknown'))

    #OneHotEncoder for Multi-Class Columns
    oh_encoder = OneHotEncoder(handle_unknown="ignore")
    oh_encoder.fit(working_train_df[multi_cols])
    train_enc_cols = oh_encoder.transform(working_train_df[multi_cols]).toarray()
    val_enc_cols = oh_encoder.transform(working_val_df[multi_cols]).toarray()
    test_enc_cols = oh_encoder.transform(working_test_df[multi_cols]).toarray()

    working_train_df.drop(columns=multi_cols, axis=1, inplace=True)
    working_val_df.drop(columns=multi_cols, axis=1, inplace=True)
    working_test_df.drop(columns=multi_cols, axis=1, inplace=True)

    working_train_df = np.concatenate([working_train_df.to_numpy(), train_enc_cols],axis=1)
    working_val_df = np.concatenate([working_val_df.to_numpy(), val_enc_cols],axis=1)
    working_test_df = np.concatenate([working_test_df.to_numpy(), test_enc_cols],axis=1)

    #Impute Data for Columns with Missing Data
    imputer=SimpleImputer(missing_values=np.nan,strategy="median")
    imputer.fit(working_train_df)
    train=imputer.transform(working_train_df)
    val=imputer.transform(working_val_df)
    test=imputer.transform(working_test_df)

    #Scale Data
    scaler=MinMaxScaler(feature_range=(0,1))
    scaler.fit(train)
    train=scaler.transform(train)
    val=scaler.transform(val)
    test=scaler.transform(test)

    return train,val,test