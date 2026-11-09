"""Strict inference boundary and content-addressed provenance."""
import csv
import hashlib
import io
from datetime import datetime, timezone
import numpy as np
import pandas as pd
from .data import DataValidationError

UNKNOWN = '미확인'


def sha256_bytes(payload):
    return hashlib.sha256(payload).hexdigest()


def validate_inputs(frame, features):
    if frame.empty:
        raise DataValidationError('CSV 파일에 데이터 행이 없습니다.')
    if frame.columns.duplicated().any():
        raise DataValidationError('중복 컬럼명이 있습니다.')
    missing = sorted(set(features) - set(frame.columns))
    if missing:
        raise DataValidationError('필수 컬럼 누락: ' + ', '.join(missing))
    result = frame.copy()
    for col in features:
        numeric = pd.to_numeric(result[col], errors='coerce')
        invalid = ~np.isfinite(numeric.to_numpy(dtype=float))
        if invalid.any():
            rows = (np.flatnonzero(invalid)[:5] + 1).tolist()
            raise DataValidationError(f'{col}: 숫자가 아니거나 결측·무한값인 데이터 행 {rows}')
        result[col] = numeric
    if 'label' in result:
        numeric = pd.to_numeric(result.label, errors='coerce')
        bad = (result.label.notna() & numeric.isna()) | (numeric.notna() & ~numeric.isin([0, 1]))
        if bad.any():
            raise DataValidationError('label은 0(정상), 1(이상) 또는 빈 값이어야 합니다.')
        result['label'] = numeric
    return result


def prepare_csv(payload, features):
    if not payload:
        raise DataValidationError('CSV 파일을 선택하세요. 빈 파일은 사용할 수 없습니다.')
    try:
        decoded = payload.decode('utf-8-sig')
        rows = list(csv.reader(io.StringIO(decoded)))
        if not rows or len(rows[0]) != len(set(rows[0])):
            raise DataValidationError('CSV 헤더가 없거나 중복 컬럼명이 있습니다.')
        if any(len(row) != len(rows[0]) for row in rows[1:]):
            raise DataValidationError('CSV 행의 컬럼 수가 헤더와 다릅니다. 빈 행도 제거하세요.')
        frame = pd.read_csv(io.StringIO(decoded), low_memory=False)
    except DataValidationError:
        raise
    except Exception as exc:
        raise DataValidationError(f'CSV를 읽을 수 없습니다. UTF-8 CSV 형식을 확인하세요: {exc}') from exc
    frame = validate_inputs(frame, features)
    # Never trust provenance columns supplied by the upload itself.
    frame['_source_sha256'] = sha256_bytes(payload)
    frame['_source_row_id'] = [f'csv-record:{i}' for i in range(1, len(frame) + 1)]
    return frame


def prediction_record(sample, bundle, score, threshold):
    from .risk import risk_metadata
    if not np.isfinite(score) or not 0 <= score <= 1 or not np.isfinite(threshold) or not 0 <= threshold <= 1:
        raise DataValidationError('예측 점수와 임계값은 0~1의 유한한 값이어야 합니다.')
    return dict(source_sha256=sample.get('_source_sha256', UNKNOWN),
                source_row_id=sample.get('_source_row_id', UNKNOWN),
                model_version=bundle.get('_artifact_version', UNKNOWN),
                threshold=float(threshold), prediction='이상' if score >= threshold else '정상',
                score=float(score), recorded_at=datetime.now(timezone.utc).isoformat(),
                **risk_metadata(bundle,score))
