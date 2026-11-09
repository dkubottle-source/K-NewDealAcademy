"""Reproducible supervised Random Forest baseline for CAD-TCM."""
from datetime import datetime, timezone
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, confusion_matrix, brier_score_loss)
from .config import FEATURES, SEED
from .data import validate_and_prepare


def split_supervised(df: pd.DataFrame):
    """Honor official train/test only if both are supervised binary data.

    CAD-TCM is a continual anomaly-detection benchmark; its official train split
    may contain only normal data. In that case, a clearly identified, independent
    stratified random split is used for this DIFFERENT supervised RF baseline.
    """
    if 'task_split' in df:
        names = df['task_split'].astype(str).str.lower().str.strip()
        is_train = names.eq('train')
        is_test = names.eq('test')
        if (is_train.sum() > 20 and is_test.sum() > 20
            and df.loc[is_train, 'label'].nunique() == 2
            and df.loc[is_test, 'label'].nunique() == 2):
            return df.loc[is_train].copy(), df.loc[is_test].copy(), '원본 task_split(train/test) 사용'
    train, test = train_test_split(df, test_size=0.2, random_state=SEED,
                                   stratify=df['label'])
    return train.copy(), test.copy(), (
        '지도학습용 별도 층화 무작위 분할(80/20). '
        'CAD-TCM의 지속학습 공식 벤치마크와 직접 비교하면 안 됩니다.'
    )


def train_rf(raw: pd.DataFrame, source_kind: str = 'cad_tcm'):
    df = validate_and_prepare(raw)
    if source_kind == 'cad_tcm' and 'task_id' in df and set(df.task_id.unique()) == set(range(8)):
        held_out = df.task_id.isin([6, 7])
        train, test = df.loc[~held_out], df.loc[held_out]
        split_note = '지도학습 태스크 홀드아웃 · 학습 0~5 / 평가 6~7. 공식 지속학습 분할과 별도.'
    else:
        train, test, split_note = split_supervised(df)
    if source_kind == 'synthetic_demo' and split_note == '원본 task_split(train/test) 사용':
        split_note = '연습용 생성 데이터의 train/test 태그 사용 (CAD-TCM 원본 분할 아님)'
    if train['label'].nunique() < 2 or test['label'].nunique() < 2:
        raise ValueError('훈련·평가 데이터에 정상과 이상이 모두 필요합니다.')
    pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median', keep_empty_features=True)),
        ('forest', RandomForestClassifier(
            n_estimators=140, max_depth=18, min_samples_leaf=2,
            class_weight='balanced_subsample', max_features='sqrt',
            random_state=SEED, n_jobs=-1,
        )),
    ])
    pipeline.fit(train[FEATURES], train['label'])
    proba = pipeline.predict_proba(test[FEATURES])[:, list(pipeline.classes_).index(1)]
    prediction = (proba >= 0.5).astype(int)
    y_true = test['label'].to_numpy()
    forest = pipeline.named_steps['forest']
    importance = pd.Series(forest.feature_importances_, index=FEATURES).sort_values(ascending=False)
    normal = train[train['label'] == 0]
    quantiles = normal[FEATURES].quantile([0.05, 0.25, 0.5, 0.75, 0.95]).T
    quantiles.columns = ['p05', 'q25', 'median', 'q75', 'p95']
    ref_stats = quantiles.astype(float).round(8).to_dict(orient='index')
    report = {
        'source_kind': source_kind,
        'training_date_utc': datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC'),
        'rows_after_validation': int(len(df)),
        'n_features': len(FEATURES),
        'train_count': int(len(train)), 'test_count': int(len(test)),
        'train_anomaly_ratio': float(train['label'].mean()),
        'test_anomaly_ratio': float(test['label'].mean()),
        'split_method': split_note,
        'accuracy': float(accuracy_score(y_true, prediction)),
        'precision': float(precision_score(y_true, prediction, zero_division=0)),
        'recall': float(recall_score(y_true, prediction, zero_division=0)),
        'f1': float(f1_score(y_true, prediction, zero_division=0)),
        'roc_auc': float(roc_auc_score(y_true, proba)),
        'brier_score': float(brier_score_loss(y_true, proba)),
        'confusion_matrix': confusion_matrix(y_true, prediction, labels=[0, 1]).tolist(),
        'feature_importance': {key: float(val) for key, val in importance.items()},
        'threshold_used_for_metrics': 0.5,
        'notices': [
            '5스탠드 합성 공정 데이터 기반 연구용 모델이며 실설비 진단 보장이 아닙니다.',
            '변수 중요도는 전체 모델 수준이며 개별 샘플의 고장 원인을 증명하지 않습니다.',
            '이상 확률은 비보정 RF 점수이며 실제 현장 고장 발생 확률과 같지 않을 수 있습니다.',
        ],
    }
    bundle = {'pipeline': pipeline, 'features': list(FEATURES),
              'source_kind': source_kind, 'normal_stats': ref_stats,
              'report': report}
    eval_df = test.loc[:, [c for c in ('task_id','task_split','label') if c in test]].copy()
    eval_df['predicted_label'] = prediction
    eval_df['anomaly_score'] = proba
    return bundle, report, eval_df


def predict_row(bundle: dict, row: pd.DataFrame) -> float:
    from .provenance import validate_inputs
    ordered = validate_inputs(row, bundle['features']).loc[:, bundle['features']]
    pipeline = bundle['pipeline']
    proba = pipeline.predict_proba(ordered)
    classes = list(pipeline.classes_)
    return float(proba[0, classes.index(1)])
