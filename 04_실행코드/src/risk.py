"""Empirical triage levels, separate from the binary detection threshold."""
import json
import numpy as np
from .config import ROOT

REGISTRY = ROOT/'reports/phase4/model_registry.json'


def read_registry():
    if not REGISTRY.exists():
        return {}
    return json.loads(REGISTRY.read_text(encoding='utf-8'))


def attach_policy(bundle):
    if 'risk_policy' not in bundle:
        policy=read_registry().get('legacy_risk_policies',{}).get(bundle.get('_artifact_version'))
        if policy:
            bundle['risk_policy']=policy
    return bundle


def risk_level(score,policy):
    if not policy:
        return '미확인'
    caution,warning=policy['caution_threshold'],policy['warning_threshold']
    if not np.isfinite([score,caution,warning]).all() or not 0<=score<=1 or not 0<=caution<=warning<=1:
        raise ValueError('위험도 점수·임계값을 확인하세요.')
    return '경고' if score>=warning else '주의' if score>=caution else '정상'


def risk_metadata(bundle,score):
    policy=bundle.get('risk_policy')
    return dict(risk_level=risk_level(score,policy),
        risk_caution_threshold=policy['caution_threshold'] if policy else '미확인',
        risk_warning_threshold=policy['warning_threshold'] if policy else '미확인',
        risk_policy_version=policy.get('version','미확인') if policy else '미확인',
        model_features=json.dumps(bundle['features'],ensure_ascii=False))


def render_risk(sample,bundle,score):
    import streamlit as st
    from .insights import get_inspection_candidates
    policy=bundle.get('risk_policy')
    level=risk_level(score,policy)
    st.metric('연구용 위험도',level)
    st.metric('선택 모델 이상 점수',f'{score:.6g}')
    if policy:
        st.caption(f"주의 ≥ {policy['caution_threshold']:.6g} / 경고 ≥ {policy['warning_threshold']:.6g}. 개발 태스크 OOF에서 Recall 0.90 / 0.70 목표로 정한 점수 구간입니다.")
    else:
        st.info('이 모델에 검증된 위험도 구간이 없어 미확인으로 표시합니다.')
    st.caption('정상 구간도 실제 설비의 안전을 보장하지 않습니다. 주의·경고는 운전 조건과 센서 확인을 돕는 연구용 분류이며 고장 확률이나 안전 기준이 아닙니다. 이진 판정 임계값과 별도로 유지됩니다.')
    candidates=get_inspection_candidates(bundle,sample,top_n=5)
    if not candidates.empty:
        st.dataframe(candidates,hide_index=True,width='stretch')
