import pandas as pd
from src.data_processing import load_raw, clean_data, train_test_split_data

def test_clean_data_no_nulls():
    df_raw = load_raw()
    df_clean = clean_data(df_raw)
    assert df_clean.isnull().sum().sum() == 0

def test_train_test_split_shapes():
    df_raw = load_raw()
    df_clean = clean_data(df_raw)
    X_train, X_test, y_train, y_test = train_test_split_data(df_clean)
    assert len(X_train) > 0
    assert len(X_test) > 0
    assert len(X_train) == len(y_train)
    assert len(X_test) == len(y_test)
