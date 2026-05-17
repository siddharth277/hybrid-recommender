"""
data_preprocessing.py - Cleans, normalizes, and encodes the dataset
before it is passed to adapt_data() and the recommender models.
Should be called before running either filtering method.
"""

import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder


def handle_missing_values(df):
    """Fill missing values. Numeric columns get median, text gets empty string."""
    df = df.dropna(how='all')
    for col in df.columns:
        if df[col].dtype in ['float64', 'int64']:
            df[col] = df[col].fillna(df[col].median())
        else:
            df[col] = df[col].fillna('')
    return df


def remove_duplicates(df):
    """Remove duplicate rows based on user+item columns if available."""
    user_col = next(
        (c for c in df.columns if any(k in c.lower() for k in ['user_id', 'user', 'reviewer'])),
        None
    )
    item_col = next(
        (c for c in df.columns if any(k in c.lower() for k in ['book_id', 'movie_id', 'product_id', 'item_id', 'asin'])),
        None
    )
    if user_col and item_col:
        df = df.drop_duplicates(subset=[user_col, item_col])
    else:
        df = df.drop_duplicates()
    return df


def normalize_ratings(df):
    """Normalize any rating column from its original scale to 0-1."""
    rating_col = next(
        (c for c in df.columns if any(k in c.lower() for k in ['rating', 'score', 'stars'])),
        None
    )
    if rating_col:
        df[rating_col] = pd.to_numeric(df[rating_col], errors='coerce')
        df = df.dropna(subset=[rating_col])
        scaler = MinMaxScaler(feature_range=(0, 1))
        df['rating_normalized'] = scaler.fit_transform(df[[rating_col]])
    return df


def encode_categorical(df):
    """Label encode short categorical columns. Skips long text like descriptions."""
    le = LabelEncoder()
    skip_keywords = ['description', 'review', 'summary', 'text', 'comment', 'overview', 'combined']
    for col in df.select_dtypes(include='object').columns:
        is_text = any(kw in col.lower() for kw in skip_keywords)
        avg_len = df[col].astype(str).str.len().mean()
        if not is_text and avg_len < 100:
            df[f'{col}_encoded'] = le.fit_transform(df[col].astype(str))
    return df


def preprocess(df):
    """
    Full preprocessing pipeline. Returns a clean DataFrame
    ready for adapt_data() and model input.
    """
    df = handle_missing_values(df)
    df = remove_duplicates(df)
    df = normalize_ratings(df)
    df = encode_categorical(df)
    return df