"""Data validation. Metadata is NOT used as predictive input."""
import pandas as pd
import numpy as np
from .config import FEATURES

class DataValidationError(ValueError):
    pass


def load_table(path):
    try:
        df = pd.read_csv(path, low_memory=False)
    except Exception as exc:
        raise DataValidationError(f'CSV 파일을 읽을 수 없습니다: {exc}') from exc
    if df.empty:
        raise DataValidationError('CSV 파일이 비어 있습니다.')
    return df


def validate_and_prepare(raw: pd.DataFrame) -> pd.DataFrame:
    missing = sorted(set(FEATURES + ['label']) - set(raw.columns))
    if missing:
        raise DataValidationError('필수 컬럼 누락: ' + ', '.join(missing))
    df = raw.copy()
    for col in FEATURES:
        df[col] = pd.to_numeric(df[col], errors='coerce').replace([np.inf, -np.inf], np.nan)
    df['label'] = pd.to_numeric(df['label'], errors='coerce')
    df = df.dropna(subset=['label']).copy()
    if df.empty or not df['label'].isin([0, 1]).all():
        raise DataValidationError('label에는 0(정상)과 1(이상)만 있어야 합니다.')
    df['label'] = df['label'].astype('int8')
    if df['label'].nunique() != 2:
        raise DataValidationError('Random Forest 이진 분류에는 정상(0)과 이상(1)이 모두 필요합니다.')
    # Identical feature vectors may appear in multiple rows; do not allow the same vector in both sets.
    # Conflicting labels are removed rather than silently choosing one label.
    conflict_counts = df.groupby(FEATURES, dropna=False, sort=False)['label'].nunique()
    if (conflict_counts > 1).any():
        conflict_hashes = pd.util.hash_pandas_object(df[FEATURES], index=False)
        conflict_df = df.assign(_feature_hash=conflict_hashes)
        bad = conflict_df.groupby('_feature_hash')['label'].transform('nunique') > 1
        df = df.loc[~bad].copy()
    df = df.drop_duplicates(subset=FEATURES, keep='first').reset_index(drop=True)
    if df['label'].nunique() != 2:
        raise DataValidationError('중복 제거 이후 한 종류의 라벨만 남았습니다.')
    return df
