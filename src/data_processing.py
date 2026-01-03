import pandas as pd
from sklearn.model_selection import train_test_split

TARGET_COL = "target"

def load_raw(path: str = "data/heart.csv") -> pd.DataFrame:
    df = pd.read_csv(path)
    return df

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.dropna()
    return df

def train_test_split_data(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    df_raw = load_raw()
    df_clean = clean_data(df_raw)
    df_clean.to_csv("data/heart_clean.csv", index=False)
