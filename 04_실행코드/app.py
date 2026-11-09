"""Cold rolling equipment anomaly screening dashboard: Streamlit."""
import altair as alt
from sklearn.metrics import roc_curve, precision_recall_curve, average_precision_score
from src.monitoring import render_monitoring, render_analysis, render_replay, render_history
import joblib
import uuid
from pathlib import Path
from src.provenance import prepare_csv, sha256_bytes, prediction_record
from src.monitoring import record_detection
import numpy as np
import pandas as pd
import streamlit as st
from src.config import ROOT, DATA_DIR, MODEL_DIR, FEATURES, GROUPS, korean_feature, DATASET_URL
from src.modeling import predict_row
from src.insights import get_inspection_candidates
from src.risk import attach_policy, read_registry, render_risk, REGISTRY
from src.phase4_dashboard import render_comparison, render_task_performance

st.set_page_config(page_title='냉간압연 설비 이상 탐지', page_icon='🏭', layout='wide', initial_sidebar_state='expanded')
st.markdown("""
<style>
.block-container {padding-top: 3.8rem; max-width: 1500px;}
#MainMenu {visibility: hidden;}
[data-testid='stMetric'] {background: #13283d; border: 1px solid #28465c;
  border-radius: 13px; padding: 13px 16px;}
[data-testid='stMetricValue'] {font-size: 2rem;}
.small-note {color: #b7c8d5; font-size: .87rem;}
.hero {background: linear-gradient(110deg,#10293e,#183b48); padding: 18px 22px;
 border-radius: 16px; border: 1px solid #335366; margin-bottom: 1.2rem;}
.hero h2 {margin-bottom: 5px; color: #eafbff;}
</style>
""", unsafe_allow_html=True)

@st.cache_data(show_spinner=False)
def load_csv(file_path: str, modified: float):
    return pd.read_csv(file_path, low_memory=False)

@st.cache_resource(show_spinner=False)
def load_bundle(file_path: str, modified: float, registry_modified: float = 0):
    # Load only artifacts generated locally by this project. Never unpickle arbitrary uploads.
    bundle = joblib.load(file_path)
    bundle['_artifact_version'] = 'sha256:' + sha256_bytes(Path(file_path).read_bytes())
    return attach_policy(bundle)


@st.cache_data(show_spinner=False)
def load_prepared(file_path, modified, features):
    return prepare_csv(Path(file_path).read_bytes(), features)


def choose_source():
    sources = []
    cad_model = MODEL_DIR / 'cad_tcm_rf.joblib'
    cad_data = DATA_DIR / 'cad_tcm.csv'
    demo_model = MODEL_DIR / 'demo_rf.joblib'
    demo_data = DATA_DIR / 'synthetic_demo.csv'
    if cad_model.exists() and cad_data.exists():
        sources.append(('CAD-TCM 원본', cad_model, cad_data, 'cad_tcm'))
    candidate = MODEL_DIR / 'cad_tcm_candidate_rf.joblib'
    if candidate.exists() and cad_data.exists():
        sources.append(('CAD-TCM 튜닝 후보', candidate, cad_data, 'cad_tcm'))
    if demo_model.exists() and demo_data.exists():
        sources.append(('연습용 합성 데이터', demo_model, demo_data, 'synthetic_demo'))
    for family, info in read_registry().get('candidates', {}).items():
        path=MODEL_DIR/info['file']
        if path.exists() and cad_data.exists():
            sources.append((f'4차 연구 후보 · {family}',path,cad_data,'cad_tcm'))
    if not sources:
        st.error('학습 모델이 없습니다. README의 설치·학습 단계를 먼저 실행하세요.')
        st.code('python -m scripts.make_demo_data\npython -m scripts.train --data data/synthetic_demo.csv --kind synthetic_demo\npython -m streamlit run app.py')
        st.stop()
    selected_name = st.sidebar.selectbox('데이터 및 학습 모델', [s[0] for s in sources])
    return next(s for s in sources if s[0] == selected_name)


def stand_comparison(sample, bundle, key):
    prefix = GROUPS[key]
    ids = range(0,6) if prefix == 'tension' else range(1,6)
    rows = []
    for i in ids:
        col = f'{prefix}_{i}'
        if col in sample.index:
            stats = bundle.get('normal_stats',{}).get(col,{})
            rows.append({'위치':('입측(0)' if i==0 else f'{i}번'),
                         '선택 사례':float(sample[col]),
                         '정상 중앙값':float(stats.get('median',np.nan))})
    frame = pd.DataFrame(rows).set_index('위치')
    units = {'force': 'N', 'torque': 'N·m', 'tension': 'N', 'gap': 'mm',
             'motor_power': 'kW', 'roll_speed': '원자료 단위 미기재', 'reduction': '무차원'}
    st.bar_chart(frame, stack=False, y_label=f'{key} ({units[prefix]})')
    st.caption('원본 데이터 수치 그대로 표시. 정상 중앙값은 학습 데이터의 정상 사례에서 계산한 참고값입니다.')


def main():
    st.markdown('<div class="hero"><h2>🏭 AI 기반 냉간압연 설비 이상 탐지</h2><div class="small-note">5스탠드 운전 데이터 분석 · 연구 모델 비교 · 점검 참고 정보</div></div>', unsafe_allow_html=True)
    st.sidebar.header('분석 설정')
    name, model_file, data_file, source_kind = choose_source()
    try:
        bundle = load_bundle(str(model_file), model_file.stat().st_mtime, REGISTRY.stat().st_mtime if REGISTRY.exists() else 0)
        df = load_prepared(str(data_file), data_file.stat().st_mtime, tuple(bundle['features']))
    except Exception as exc:
        st.error(f'데이터·모델 로딩 실패. 실행 경로와 파일을 확인하세요. {exc}')
        st.stop()
    demo = source_kind == 'synthetic_demo'
    if demo:
        st.warning('**연습 모드** — 현재 데이터·모델은 인위적으로 만든 합성 예제입니다. 실제 CAD-TCM의 학습 결과나 성능이 아닙니다.', icon='🧪')
    else:
        st.info('**CAD-TCM 연구 모드** — 물리 모델로 만든 합성 산업 데이터에 기반한 연구용 분류 결과입니다. 실공장 연결·고장 확정 기능은 없습니다.', icon='ℹ️')

    input_mode = st.sidebar.radio('예측 데이터 출처', ['기본 데이터', 'CSV 업로드'])
    uploaded = st.sidebar.file_uploader('추가 CSV 예측 (선택)', type=['csv'])
    if input_mode == 'CSV 업로드' or uploaded is not None:
        if uploaded is None:
            st.sidebar.warning('CSV 파일을 선택하세요. 예측과 점검 기록 저장이 차단되었습니다.')
            st.stop()
        try:
            df = prepare_csv(uploaded.getvalue(), bundle['features'])
        except Exception as exc:
            st.sidebar.error(f'CSV 검증 실패: {exc}')
            st.error('업로드 검증에 실패하여 AI 예측 및 점검 기록 저장을 차단했습니다.')
            st.stop()
        st.sidebar.success(f'CSV 검증 성공: {len(df):,}개 사례')
        st.caption('원본 행 식별자는 헤더를 제외한 1부터 시작하는 CSV 레코드 번호입니다. 단위와 운전 조건은 별도 확인하세요.')
    if not set(bundle['features']).issubset(df.columns):
        st.error('현재 데이터와 저장된 모델의 입력 컬럼이 다릅니다.')
        st.stop()
    threshold = st.sidebar.slider('이상 판정 임계값', min_value=0.0, max_value=1.0,
                                  value=float(bundle['report']['threshold_used_for_metrics']), step=0.001,
                                  key='binary_cutoff_'+bundle['_artifact_version'])
    if bundle.get('risk_policy'):
        cutoff_key='binary_cutoff_'+bundle['_artifact_version']
        st.sidebar.button('주의 cutoff를 이진 기준에 적용 (연구용)',
            on_click=lambda:st.session_state.update({cutoff_key:float(bundle['risk_policy']['caution_threshold'])}))
        st.sidebar.button('모델 저장 이진 기준으로 복원',
            on_click=lambda:st.session_state.update({cutoff_key:float(bundle['report']['threshold_used_for_metrics'])}))
    st.sidebar.caption('임계값 변경은 현재 화면의 판정만 바꿉니다. 모델 학습과 저장된 평가 점수는 변하지 않습니다.')
    st.sidebar.caption(f"저장된 이진 기준값: {bundle['report']['threshold_used_for_metrics']:.8g} / 현재: {threshold:.8g}")
    st.sidebar.markdown(f'[CAD-TCM 데이터셋 원본]({DATASET_URL})')

    pages = ['메인 대시보드', '공정 데이터 분석', 'AI 이상 탐지', '점검 지원', 'AI 모델 성능', '설비 모니터링', '운전 시뮬레이션', '이상 이력', 'AI 모델 비교', '위험도 분석', '태스크별 성능']
    page = st.sidebar.radio('화면 선택', pages, key='page_choice')
    report = bundle['report']
    if page in pages[1:4] + ['설비 모니터링', '위험도 분석']:
        with st.sidebar:
            if 'label' in df.columns:
                mode = st.radio('사례 필터 (실제 라벨 기준)', ['전체', '정상(0)', '이상(1)'], horizontal=True)
                if mode == '정상(0)':
                    filtered = df.loc[df['label'].eq(0)]
                elif mode == '이상(1)':
                    filtered = df.loc[df['label'].eq(1)]
                else:
                    filtered = df
            else:
                filtered = df
            if filtered.empty:
                st.warning('해당 조건의 데이터가 없습니다.')
                st.stop()
            position = st.number_input('선택할 사례 순번 (0부터 시작)',
                                       min_value=0, max_value=len(filtered)-1, value=0, step=1)
            sample_id = filtered.index[int(position)]
            sample = df.loc[sample_id]
    if page == '설비 모니터링':
        render_monitoring(sample, bundle, threshold)
    if page == '운전 시뮬레이션':
        if demo:
            st.info('CAD-TCM 모델을 선택하면 원본 재생을 시작할 수 있습니다.')
        elif not (DATA_DIR / 'cad_tcm.csv').exists():
            st.error('CAD-TCM 원본 파일을 먼저 내려받으세요.')
        else:
            original = load_prepared(str(DATA_DIR/'cad_tcm.csv'), (DATA_DIR/'cad_tcm.csv').stat().st_mtime, tuple(bundle['features']))
            render_replay(original, bundle, threshold, model_file.name + ':' + str(model_file.stat().st_mtime))
    if page == '이상 이력':
        render_history()
    if page == 'AI 모델 비교':
        render_comparison()
    if page == '태스크별 성능':
        render_task_performance(df)
    if page == '위험도 분석':
        score = predict_row(bundle,pd.DataFrame([sample]))
        render_risk(sample,bundle,score)
        render_analysis(sample,sample_id,bundle,threshold,name,model_file.name)
    if page == 'AI 이상 탐지':
        prob = predict_row(bundle, pd.DataFrame([sample]))
        flag = prob >= threshold
        prediction = prediction_record(sample, bundle, prob, threshold)
        record_detection(str(uuid.uuid4()), name + (' · 업로드' if uploaded is not None else ''),
                         model_file.name, sample_id, None, prob, threshold, sample, bundle)
        st.write(f'**사례 번호:** {sample_id} / **원본 행 수:** {len(df):,}')
        if 'task_name' in sample and pd.notna(sample['task_name']):
            st.caption('공정 태스크: ' + str(sample['task_name']))
        c1, c2 = st.columns(2)
        c1.metric('AI 예측 상태', '🔴 이상' if flag else '🟢 정상')
        c2.metric('AI 이상 점수', f'{prob*100:.1f}%')
        st.progress(min(max(prob, 0), 1))
        render_risk(sample,bundle,prob)
        render_analysis(sample, sample_id, bundle, threshold, name + (' · 업로드' if uploaded is not None else ''), model_file.name)
        if 'label' in sample and pd.notna(sample['label']):
            st.caption('데이터셋 실제 라벨: ' + ('이상(1)' if int(sample['label']) else '정상(0)') + ' · AI 예측과 실제 라벨이 다를 수 있습니다.')
        st.caption('이상 점수는 선택한 연구 모델의 분류 점수이며 실제 설비 고장 확률로 해석할 수 없습니다.')
        st.download_button('선택 사례 예측 결과 CSV 저장',
            data=pd.DataFrame([dict(sample_index=sample_id, **prediction)]).to_csv(index=False).encode('utf-8-sig'),
            file_name='selected_prediction.csv', mime='text/csv')

    elif page == '공정 데이터 분석':
        st.caption(f"선택 사례 {sample_id:,}번")
        st.subheader('스탠드별 운전 변수')
        categories = st.multiselect('변수 종류', list(GROUPS), default=['압연력','롤 토크','장력','롤 속도'])
        if categories:
            for i in range(0, len(categories), 2):
                cols = st.columns(2, gap='large')
                for j, group in enumerate(categories[i:i+2]):
                    with cols[j]:
                        st.markdown(f'**{group}**')
                        stand_comparison(sample, bundle, group)
        st.dataframe(sample[bundle['features']].rename('선택 사례 값'))
    elif page == '점검 지원':
        st.caption(f'선택 사례 {sample_id:,}번')
        render_analysis(sample, sample_id, bundle, threshold, name + (' · 업로드' if uploaded is not None else ''), model_file.name)
        st.info('저장한 점검 기록의 상태·확인 결과·메모는 이상 이력 화면에서 수정할 수 있습니다.')
        st.subheader('점검 참고 항목')
        candidate = get_inspection_candidates(bundle, sample, top_n=7)
        if candidate.empty:
            st.warning('비교할 정상 참조 통계가 없습니다.')
        else:
            unusual = candidate.loc[candidate['정상 참고범위 벗어남']]
            if len(unusual):
                st.warning(f'정상 학습 데이터의 5~95% 참고범위를 벗어난 입력 변수 {len(unusual)}개가 상위 목록에 있습니다.')
            else:
                st.info('상위 참고 항목에서 정상 5~95% 범위를 벗어난 변수는 없습니다.')
            st.dataframe(candidate[['변수', '선택 사례 값', '정상 기준 중앙값', '중앙값 대비 IQR 편차', '정상 참고범위 벗어남']],
                         hide_index=True, width="stretch")
        st.caption('정상 분포와의 차이로 정렬한 점검 후보입니다. 고장 발생 위치·원인 또는 이 예측의 직접적 설명을 의미하지 않습니다.')
        st.subheader('원본 데이터 변수 분포')
        st.caption('아래 분포는 대시보드에서 현재 선택한 데이터의 통계입니다. 시간 순서 그래프가 아닙니다.')
        choice = st.selectbox('비교할 변수', bundle['features'],
                              format_func=lambda x: f'{korean_feature(x)} ({x})')
        vals = df[choice].dropna()
        if len(vals):
            edges = np.histogram_bin_edges(vals, bins=25)
            groups = df.groupby('label') if 'label' in df else [('전체', df)]
            hist = pd.DataFrame({'구간 중앙값': (edges[:-1] + edges[1:]) / 2})
            for label, group in groups:
                counts, _ = np.histogram(group[choice].dropna(), bins=edges)
                hist[{0: '정상', 1: '이상'}.get(label, '전체')] = 100 * counts / max(1, counts.sum())
            st.caption(f'평균 {vals.mean():,.4g} · 중앙값 {vals.median():,.4g} · 표준편차 {vals.std():,.4g}')
            st.bar_chart(hist.set_index('구간 중앙값'), stack=False, y_label='클래스 내 비율 (%)')
    if page in ['점검 지원', 'AI 모델 성능']:
        st.markdown('#### AI 변수 중요도 (전체 모델 기준)')
        importances = bundle['report']['feature_importance']
        imp = pd.DataFrame([{'변수':f'{korean_feature(k)} ({k})', '중요도':v}
                            for k,v in list(importances.items())[:20]],columns=['변수','중요도']).set_index('변수')
        if not imp.empty:
            st.bar_chart(imp)
        else:
            st.info('이 모델은 불순도 기반 변수 중요도를 제공하지 않습니다. 정상 분포 편차를 점검 참고 정보로 확인하세요.')
        st.caption('트리 모델은 불순도 중요도, LogisticRegression은 표준화 계수 절댓값입니다. 변수 간 상관관계의 영향을 받으며 고장 원인을 증명하지 않습니다.')

    if page == 'AI 모델 성능':
        st.subheader('학습·평가 정보')
        report = bundle['report']
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric('Accuracy', f"{report['accuracy']:.3f}")
        m2.metric('Precision', f"{report['precision']:.3f}")
        m3.metric('Recall', f"{report['recall']:.3f}")
        m4.metric('F1', f"{report['f1']:.3f}")
        m5.metric('ROC-AUC', f"{report['roc_auc']:.3f}")
        st.markdown(f"**평가 기준 임계값:** {report['threshold_used_for_metrics']:.2f} · **학습 시각:** {report['training_date_utc']}")
        st.markdown(f"**분할 방식:** {report['split_method']}")
        st.caption(f"훈련 {report['train_count']:,}건 / 평가 {report['test_count']:,}건 / 입력 변수 {report['n_features']}개")
        cm = report['confusion_matrix']
        st.markdown('**혼동행렬 (행: 실제 / 열: 예측)**')
        st.dataframe(pd.DataFrame(cm, index=['실제 정상','실제 이상'],
                                  columns=['예측 정상','예측 이상']), width="stretch")
        st.write(f'평가 이상 {sum(cm[1]):,}건 중 {cm[1][0]:,}건을 정상으로 판정했습니다.')
        for notice in report.get('notices',[]):
            st.caption('• ' + notice)
        result_file = model_file.with_name(model_file.name.replace('_rf.joblib', '_test_predictions.csv'))
        comparison = ROOT / 'reports' / 'phase2' / 'comparison.csv'
        if comparison.exists() and not demo:
            st.subheader('기존 모델과 튜닝 후보 비교')
            st.dataframe(pd.read_csv(comparison), hide_index=True)
            st.caption('후보 선택과 임계값은 태스크 4~5 검증 F1로 결정했습니다. 최종 테스트 6~7의 점수는 모델 선택에 사용하지 않았습니다. 기존 모델이 기본값입니다.')
        if model_file.name.endswith('_rf.joblib') and result_file.exists():
            evaluation = load_csv(str(result_file), result_file.stat().st_mtime)
            precision, recall, _ = precision_recall_curve(evaluation.label, evaluation.anomaly_score)
            st.metric('Average Precision (AP)', f'{average_precision_score(evaluation.label, evaluation.anomaly_score):.4f}')
            pr = pd.DataFrame({'재현율': recall, '정밀도': precision, '순서': range(len(recall))})
            st.subheader('Precision–Recall 곡선')
            st.altair_chart(alt.Chart(pr).mark_line().encode(x='재현율:Q', y='정밀도:Q', order='순서:Q'), width='stretch')
            fpr, tpr, _ = roc_curve(evaluation.label, evaluation.anomaly_score)
            curve = pd.DataFrame({'위양성률': fpr, '재현율': tpr, '순서': range(len(fpr))})
            st.subheader('ROC 곡선')
            st.altair_chart(alt.Chart(curve).mark_line().encode(
                x='위양성률:Q', y='재현율:Q', order='순서:Q'), width="stretch")
            st.download_button('전체 평가 예측 CSV 다운로드', result_file.read_bytes(),
                               file_name=result_file.name, mime='text/csv')

    if page == '메인 대시보드':
        st.subheader('공정 데이터 현황')
        metrics = st.columns(4)
        metrics[0].metric('전체 사례', f'{len(df):,}건')
        labels = df.get('label', pd.Series(dtype=float)).dropna()
        for slot, value, name in [(1, 0, '정상'), (2, 1, '이상')]:
            metrics[slot].metric(name, f'{labels.eq(value).sum():,}건' if len(labels) else '라벨 없음')
        metrics[3].metric('라벨 중 이상 비율', f'{labels.mean():.2%}' if len(labels) else '라벨 없음')
        st.subheader('모델 평가 요약')
        cols = st.columns(3)
        for col, key, title in zip(cols, ['recall','f1','roc_auc'], ['이상 재현율','이상 F1','ROC-AUC']):
            col.metric(title, f'{report[key]:.3f}')
        st.caption(report['split_method'])
        st.subheader('데이터 미리보기')
        st.caption(f'전체 {len(df.columns)}열 · 결측값 {df.isna().sum().sum():,}개 · 처음 250행 표시')
        st.dataframe(df.head(250), width="stretch")
        with st.expander('컬럼 설명 보기'):
            st.dataframe(pd.DataFrame({'원본 컬럼':bundle['features'],
                                      '표시명':[korean_feature(c) for c in bundle['features']]}),
                         hide_index=True, width="stretch")
        st.caption('label(정답), task_id, task_name, task_split은 모델 입력에 사용하지 않습니다.')

    st.divider()
    st.caption('CAD-TCM: Lifelong Continual Lab / TCM 기반 합성 데이터 · CC BY 4.0 · 연구·교육용 시연. 본 화면은 PLC/실제 공장 설비에 연결되지 않습니다.')

if __name__ == '__main__':
    main()
