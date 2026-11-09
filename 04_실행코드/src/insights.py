"""Normal-baseline distance is a screening heuristic, NOT causal diagnostics."""
import numpy as np
import pandas as pd
from .config import korean_feature


def get_inspection_candidates(bundle: dict, sample: pd.Series, top_n: int = 8) -> pd.DataFrame:
    records = []
    for feat in bundle['features']:
        v = sample.get(feat)
        s = bundle['normal_stats'].get(feat, {})
        median = s.get('median', np.nan)
        q25, q75 = s.get('q25', np.nan), s.get('q75', np.nan)
        p05, p95 = s.get('p05', np.nan), s.get('p95', np.nan)
        if pd.isna(v) or pd.isna(median):
            continue
        spread = q75-q25
        if pd.isna(spread) or spread <= 0:
            spread = max(abs(median)*0.01, 1e-6)
        deviation = abs(float(v)-median)/spread
        outside = bool(float(v) < p05 or float(v) > p95)
        records.append({'변수': korean_feature(feat), '원본 컬럼': feat,
                        '선택 사례 값': float(v), '정상 기준 중앙값': median,
                        '정상 5%': p05, '정상 95%': p95,
                        '중앙값 대비 IQR 편차': round(float(deviation), 2),
                        '정상 참고범위 벗어남': outside})
    result = pd.DataFrame(records)
    if result.empty:
        return result
    result = result.sort_values(['정상 참고범위 벗어남', '중앙값 대비 IQR 편차'],
                                ascending=[False, False])
    return result.head(top_n).reset_index(drop=True)
