"""Task-separated supervised tuning. Run with python -m scripts.tune."""
import json
from datetime import datetime, timezone
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, average_precision_score, confusion_matrix,
                             precision_recall_curve)
from src.config import ROOT, FEATURES, SEED
from src.data import load_table, validate_and_prepare


def metrics(y, score, threshold):
    prediction = score >= threshold
    return dict(accuracy=float(accuracy_score(y, prediction)),
                precision=float(precision_score(y, prediction, zero_division=0)),
                recall=float(recall_score(y, prediction, zero_division=0)),
                f1=float(f1_score(y, prediction, zero_division=0)),
                roc_auc=float(roc_auc_score(y, score)),
                average_precision=float(average_precision_score(y, score)),
                confusion_matrix=confusion_matrix(y, prediction).tolist(),
                threshold_used_for_metrics=float(threshold))


def run():
    output = ROOT / 'reports' / 'phase2'
    output.mkdir(parents=True, exist_ok=True)
    df = validate_and_prepare(load_table(ROOT / 'data' / 'cad_tcm.csv'))
    fit = df[df.task_id.isin([0, 1, 2, 3])]
    valid = df[df.task_id.isin([4, 5])]
    development = df[df.task_id.isin(range(6))]
    # No test features or labels are accessed during model/threshold selection.
    process_features = [c for c in FEATURES if c.startswith(
        ('force_', 'torque_', 'tension_', 'roll_speed_', 'motor_power_', 'gap_', 'reduction_'))]
    configs = [
        dict(name='balanced_baseline', weight='balanced_subsample', depth=18, leaf=2, features=FEATURES),
        dict(name='unweighted', weight=None, depth=None, leaf=2, features=FEATURES),
        dict(name='moderate_weight', weight={0: 1, 1: 4}, depth=24, leaf=3, features=FEATURES),
        dict(name='process_only', weight=None, depth=24, leaf=2, features=process_features),
    ]
    trials = []
    for config in configs:
        features = config['features']
        pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='median', keep_empty_features=True)),
            ('forest', RandomForestClassifier(n_estimators=180, max_depth=config['depth'],
                min_samples_leaf=config['leaf'], class_weight=config['weight'],
                max_features='sqrt', random_state=SEED, n_jobs=-1))])
        pipeline.fit(fit[features], fit.label)
        score = pipeline.predict_proba(valid[features])[:, 1]
        precision, recall, thresholds = precision_recall_curve(valid.label, score)
        f1 = 2 * precision[:-1] * recall[:-1] / np.maximum(precision[:-1] + recall[:-1], 1e-12)
        threshold = float(thresholds[np.argmax(f1)])
        result = metrics(valid.label, score, threshold)
        trials.append(dict(config=config, validation=result))
        print(config['name'], result, flush=True)
    winner = max(trials, key=lambda t: (t['validation']['f1'], t['validation']['average_precision']))
    # Persist the decision before the single final test evaluation.
    (output / 'selection.json').write_text(json.dumps(dict(
        trials=trials, selected=winner, fit_tasks=[0,1,2,3], validation_tasks=[4,5],
        test_tasks=[6,7], criterion='validation maximum F1, AP tie break',
        resampling='none; compare class weights to avoid synthetic cross-task samples'),
        ensure_ascii=False, indent=2), encoding='utf8')
    config = winner['config']
    features = config['features']
    pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median', keep_empty_features=True)),
        ('forest', RandomForestClassifier(n_estimators=180, max_depth=config['depth'],
            min_samples_leaf=config['leaf'], class_weight=config['weight'],
            max_features='sqrt', random_state=SEED, n_jobs=-1))])
    pipeline.fit(development[features], development.label)
    test = df[df.task_id.isin([6,7])]
    score = pipeline.predict_proba(test[features])[:,1]
    threshold = winner['validation']['threshold_used_for_metrics']
    report = metrics(test.label, score, threshold)
    baseline = joblib.load(ROOT / 'models' / 'cad_tcm_rf.joblib')
    baseline_score = baseline['pipeline'].predict_proba(test[baseline['features']])[:,1]
    baseline_result = metrics(test.label, baseline_score, .5)
    report.update(source_kind='cad_tcm', train_count=len(development), test_count=len(test),
        n_features=len(features), training_date_utc=datetime.now(timezone.utc).isoformat(),
        split_method='학습 태스크 0~5 / 최종 평가 6~7 · 튜닝 학습 0~3 / 검증 4~5',
        feature_importance=pd.Series(pipeline.named_steps['forest'].feature_importances_,
            index=features).sort_values(ascending=False).to_dict())
    # Reference statistics cover all raw features for stand monitoring.
    normal = development[development.label.eq(0)]
    q = normal[FEATURES].quantile([.05,.25,.5,.75,.95]).T
    q.columns = ['p05','q25','median','q75','p95']
    bundle = dict(pipeline=pipeline, features=features, source_kind='cad_tcm',
                  normal_stats=q.to_dict(orient='index'), report=report)
    joblib.dump(bundle, ROOT / 'models' / 'cad_tcm_candidate_rf.joblib', compress=3)
    (ROOT / 'models' / 'cad_tcm_candidate_metrics.json').write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding='utf8')
    result = test[['task_id','task_split','label']].copy()
    result['predicted_label'] = (score >= threshold).astype(int)
    result['anomaly_score'] = score
    result.to_csv(ROOT / 'models' / 'cad_tcm_candidate_test_predictions.csv', index=False)
    comparison = pd.DataFrame([dict(model='기존 모델', **baseline_result),
                               dict(model='튜닝 후보', **metrics(test.label, score, threshold))])
    comparison.to_json(output / 'comparison.json', orient='records', force_ascii=False, indent=2)
    comparison.drop(columns='confusion_matrix').to_csv(output / 'comparison.csv', index=False, encoding='utf-8-sig')
    print(comparison.to_string(), flush=True)


if __name__ == '__main__':
    run()
