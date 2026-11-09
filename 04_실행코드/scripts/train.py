"""Run: python -m scripts.train --data data/cad_tcm.csv"""
import argparse
import json
from pathlib import Path
import joblib
from src.config import ROOT, MODEL_DIR
from src.data import load_table
from src.modeling import train_rf


def train(data: Path, output: Path, kind: str):
    print(f'데이터 읽는 중: {data}')
    raw = load_table(data)
    print(f'총 {len(raw):,}행, {len(raw.columns)}개 컬럼')
    print('Random Forest 모델 학습 및 독립 평가 중...')
    bundle, report, results = train_rf(raw, source_kind=kind)
    output.mkdir(parents=True, exist_ok=True)
    suffix = 'demo' if kind == 'synthetic_demo' else 'cad_tcm'
    model_path = output / f'{suffix}_rf.joblib'
    joblib.dump(bundle, model_path, compress=3)
    (output / f'{suffix}_metrics.json').write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    results.to_csv(output / f'{suffix}_test_predictions.csv', index=False)
    print(f'완료: {model_path}')
    print(f"정확도 {report['accuracy']:.3f} | 재현율 {report['recall']:.3f} | F1 {report['f1']:.3f} | ROC-AUC {report['roc_auc']:.3f}")
    print(f"분할 방식: {report['split_method']}")

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--data', type=Path, default=ROOT/'data'/'cad_tcm.csv')
    p.add_argument('--output', type=Path, default=MODEL_DIR)
    p.add_argument('--kind', choices=['cad_tcm','synthetic_demo'], default='cad_tcm')
    a = p.parse_args()
    if not a.data.exists():
        p.error(f'파일을 찾을 수 없습니다: {a.data}. 먼저 python -m scripts.download_data 실행하세요.')
    train(a.data, a.output, a.kind)
