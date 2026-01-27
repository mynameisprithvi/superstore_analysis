import os
import pandas as pd
import numpy as np


def load_parquet(path: str) -> pd.DataFrame:
    return pd.read_parquet(path)


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    # -------------------------------------------------------
    # 1. Drop duplicates
    # -------------------------------------------------------
    print("Duplicates BEFORE drop:", df.duplicated().sum())
    df = df.drop_duplicates()
    print("Duplicates AFTER drop:", df.duplicated().sum())

    # -------------------------------------------------------
    # 2. Fix data types
    # -------------------------------------------------------
    numeric_cols = ["sales", "quantity", "discount", "profit"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Quantity should be integer
    if "quantity" in df.columns:
        df["quantity"] = df["quantity"].astype("Int64")

    # Convert date columns
    if "order_date" in df.columns:
        df["order_date"] = pd.to_datetime(df["order_date"], errors="coerce")
    if "ship_date" in df.columns:
        df["ship_date"] = pd.to_datetime(df["ship_date"], errors="coerce")

    # -------------------------------------------------------
    # 3. Clean category strings
    # -------------------------------------------------------
    cat_cols = df.select_dtypes(include="object").columns
    for col in cat_cols:
        df[col] = df[col].astype(str).str.strip()

    # -------------------------------------------------------
    # 4. Create date features
    # -------------------------------------------------------
    if "order_date" in df.columns:
        df["order_year"] = df["order_date"].dt.year
        df["order_month"] = df["order_date"].dt.month
        df["order_day"] = df["order_date"].dt.day
        df["order_week"] = df["order_date"].dt.isocalendar().week.astype(int)
        df["order_weekday"] = df["order_date"].dt.weekday

    # -------------------------------------------------------
    # 5. Shipping duration (days)
    # -------------------------------------------------------
    if {"order_date", "ship_date"}.issubset(df.columns):
        df["shipping_time_days"] = (df["ship_date"] - df["order_date"]).dt.days

    # -------------------------------------------------------
    # 6. Profitability metrics
    # -------------------------------------------------------
    if {"profit", "sales"}.issubset(df.columns):
        df["profit_margin"] = df["profit"] / df["sales"]

    if {"profit", "quantity"}.issubset(df.columns):
        df["profit_per_unit"] = df["profit"] / df["quantity"].replace(0, np.nan)

    # -------------------------------------------------------
    # 7. Mild outlier handling (optional but useful)
    # -------------------------------------------------------
    if "profit" in df.columns:
        lower = df["profit"].quantile(0.01)
        upper = df["profit"].quantile(0.99)
        df["profit"] = df["profit"].clip(lower=lower, upper=upper)

    return df


if __name__ == "__main__":
    infile = r"D:\Data analysis BI\Project\Superstore_analysis\data\raw\superstore.parquet"
    outfile = r"D:\Data analysis BI\Project\Superstore_analysis\data\processed\superstore_processed.parquet"

    df = load_parquet(infile)
    df = preprocess(df)

    os.makedirs(os.path.dirname(outfile), exist_ok=True)
    df.to_parquet(outfile)
    print(f"Saved processed data to {outfile}")
