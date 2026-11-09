"""Stand displays, local sensitivity, replay, and persistent detection history."""
import sqlite3
from contextlib import contextmanager
import uuid
import json
from datetime import datetime, timezone
import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from .config import ROOT, korean_feature
from .modeling import predict_row
from .insights import get_inspection_candidates
from .provenance import prediction_record, validate_inputs
from .risk import risk_level, render_risk

VARIABLES = {'force': ('압연력', 'N'), 'torque': ('롤 토크', 'N·m'),
             'tension': ('출측 장력', 'N'), 'roll_speed': ('롤 속도', '단위 미기재'),
             'motor_power': ('모터 출력', 'kW'), 'gap': ('롤 갭', 'mm')}
HISTORY_DB = ROOT / 'reports' / 'detection_history.sqlite3'


@contextmanager
def history_connection():
    HISTORY_DB.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(HISTORY_DB, timeout=10)
    connection.execute('''CREATE TABLE IF NOT EXISTS detections (
        event_id TEXT PRIMARY KEY, recorded_at TEXT, source TEXT, model TEXT,
        sample_index INTEGER, step INTEGER, prediction TEXT, score REAL,
        threshold REAL, actual_label TEXT, inspection TEXT)''')
    columns = {row[1] for row in connection.execute('PRAGMA table_info(detections)')}
    for name in ('source_sha256', 'source_row_id', 'model_version', 'inspection_status',
                 'operator_result', 'inspection_note', 'inspection_updated_at'):
        if name not in columns:
            connection.execute(f"ALTER TABLE detections ADD COLUMN {name} TEXT NOT NULL DEFAULT '미확인'")
    for name, kind in [('risk_level','TEXT'),('risk_caution_threshold','REAL'),
                       ('risk_warning_threshold','REAL'),('risk_policy_version','TEXT'),('model_features','TEXT')]:
        if name not in columns:
            connection.execute(f"ALTER TABLE detections ADD COLUMN {name} {kind} NOT NULL DEFAULT '미확인'")
    connection.commit()
    try:
        with connection:
            yield connection
    finally:
        connection.close()


def record_detection(event_id, source, model, sample_id, step, score, threshold, sample, bundle):
    validate_inputs(pd.DataFrame([sample]), bundle['features'])
    provenance = prediction_record(sample, bundle, score, threshold)
    candidates = get_inspection_candidates(bundle, sample, top_n=3)
    inspection = ', '.join(candidates.get('변수', pd.Series(dtype=str)).tolist())
    label = sample.get('label')
    with history_connection() as connection:
        connection.execute('''INSERT OR IGNORE INTO detections
            (event_id, recorded_at, source, model, sample_index, step, prediction, score,
             threshold, actual_label, inspection, source_sha256, source_row_id, model_version,
             inspection_status, risk_level, risk_caution_threshold, risk_warning_threshold,
             risk_policy_version, model_features) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)''',
            (event_id, provenance['recorded_at'], source, model, int(sample_id),
             step, '이상' if score >= threshold else '정상', float(score), float(threshold),
             '' if pd.isna(label) else str(int(label)), inspection,
             provenance['source_sha256'], provenance['source_row_id'], provenance['model_version'], '미점검',
             provenance['risk_level'],provenance['risk_caution_threshold'],provenance['risk_warning_threshold'],
             provenance['risk_policy_version'],provenance['model_features']))


def update_inspection(event_id, status, result, note):
    if status not in ('미점검', '점검 중', '점검 완료'):
        raise ValueError('올바른 점검 상태를 선택하세요.')
    with history_connection() as connection:
        cursor = connection.execute('''UPDATE detections SET inspection_status=?, operator_result=?,
            inspection_note=?, inspection_updated_at=? WHERE event_id=?''',
            (status, result, note, datetime.now(timezone.utc).isoformat(), event_id))
        if cursor.rowcount != 1:
            raise ValueError('점검 기록을 찾을 수 없습니다.')


def read_history():
    with history_connection() as connection:
        return pd.read_sql_query('SELECT * FROM detections ORDER BY recorded_at DESC', connection)


def render_history():
    st.subheader('이상 감지 이력')
    st.caption('기록 시각은 프로그램 실행 시각(UTC)입니다. 데이터 측정 시각은 제공되지 않습니다. 이력은 PC에 저장됩니다.')
    frame = read_history()
    st.metric('저장된 감지·수동 점검 기록', f'{len(frame):,}건')
    st.dataframe(frame, hide_index=True, width='stretch')
    if not frame.empty:
        event = st.selectbox('수정할 점검 기록', frame.event_id.tolist())
        row = frame.set_index('event_id').loc[event]
        with st.form(f'inspection_{event}'):
            statuses = ['미확인', '미점검', '점검 중', '점검 완료']
            status = st.selectbox('점검 상태', statuses, index=statuses.index(row.inspection_status))
            result = st.text_input('작업자 확인 결과', value=row.operator_result)
            note = st.text_area('점검 메모', value=row.inspection_note)
            if st.form_submit_button('점검 정보 저장'):
                if status == '미확인':
                    st.error('점검 상태를 선택하세요.')
                else:
                    update_inspection(event, status, result, note)
                    st.rerun()
    if st.button('PDF 점검 보고서 준비'):
        try:
            from .reporting import history_pdf
            pdf = history_pdf(frame)
            st.download_button('PDF 다운로드 (최신 200건)', pdf, file_name='inspection_report.pdf', mime='application/pdf')
        except Exception as exc:
            st.error(f'PDF 생성 실패. CSV 다운로드를 이용하세요. {exc}')
    st.download_button('이력 CSV 다운로드', frame.to_csv(index=False).encode('utf-8-sig'),
                       file_name='detection_history.csv', mime='text/csv')


def render_monitoring(sample, bundle, threshold=None):
    st.subheader('5스탠드 설비 모니터링')
    st.caption('소재 진행 방향 → · 장력 i는 i번 스탠드 출측 값입니다. 입측 장력은 tension_0입니다.')
    score=predict_row(bundle,pd.DataFrame([sample]))
    render_risk(sample,bundle,score)
    st.caption('큰 편차는 점검 참고 정보이며 고장 스탠드나 원인을 확정하지 않습니다.')
    st.button('선택 사례 상세 점검 열기',
              on_click=lambda: st.session_state.update(page_choice='점검 지원'))
    selected = st.session_state.get('selected_stand', 1)
    for stand, col in enumerate(st.columns(5), start=1):
        with col, st.container(border=True):
            if st.button(f'스탠드 {stand}', key=f'stand_{stand}', width='stretch',
                         type='primary' if selected == stand else 'secondary'):
                st.session_state.selected_stand = stand
                st.rerun()
            for prefix, (label, unit) in VARIABLES.items():
                value = sample.get(f'{prefix}_{stand}', np.nan)
                number = f'{value:,.0f}' if prefix in ('force', 'torque', 'tension') else f'{value:,.3f}'
                st.markdown(f'**{label}**  \n{number} {unit}')
                stats=bundle['normal_stats'].get(f'{prefix}_{stand}',{})
                if stats and pd.notna(value):
                    spread=max(stats['q75']-stats['q25'],abs(stats['median'])*.01,1e-6)
                    deviation=abs(value-stats['median'])/spread
                    if value<stats['p05'] or value>stats['p95']:
                        st.warning(f'참고 구간 이탈 · IQR 편차 {deviation:.2f}')
    selected = st.session_state.get('selected_stand', 1)
    st.subheader(f'스탠드 {selected} 상세 비교')
    st.caption('정상 학습 데이터 5~95% 참고 구간과 중앙값을 표시합니다. 운전 조건을 합산한 통계 구간입니다.')
    for prefix, (label, unit) in VARIABLES.items():
        feature = f'{prefix}_{selected}'
        stats = bundle['normal_stats'][feature]
        frame = pd.DataFrame([dict(variable=label, low=stats['p05'], high=stats['p95'],
                                   median=stats['median'], value=sample[feature])])
        chart = alt.Chart(frame)
        band = chart.mark_bar(opacity=.25, size=20).encode(x='low:Q', x2='high:Q', y='variable:N')
        mid = chart.mark_tick(color='#23c4bd', thickness=3, size=28).encode(x='median:Q', y='variable:N')
        point = chart.mark_point(color='#ffbc66', size=120, filled=True).encode(x='value:Q', y='variable:N',
            tooltip=['value:Q','low:Q','high:Q','median:Q'])
        st.altair_chart((band + mid + point).properties(height=75, title=f'{label} ({unit})'), width='stretch')
    st.caption(f"입측 장력 tension_0 · {sample.get('tension_0', np.nan):,.4g} N · 단위 출처 https://zenodo.org/records/11469702")


def local_sensitivity(bundle, sample):
    """One-feature replacement sensitivity relative to the normal-training median."""
    features = bundle['features']
    rows = pd.DataFrame([sample[features]] * len(features)).reset_index(drop=True)
    for i, feature in enumerate(features):
        rows.loc[i, feature] = bundle['normal_stats'][feature]['median']
    scores = bundle['pipeline'].predict_proba(rows)[:, 1]
    original = predict_row(bundle, pd.DataFrame([sample]))
    return pd.DataFrame({'변수': [korean_feature(f) for f in features],
                         '점수 변화': original - scores}).sort_values('점수 변화', ascending=False)


def render_analysis(sample, sample_id, bundle, threshold, source, model):
    candidates = get_inspection_candidates(bundle, sample, top_n=10)
    st.subheader('정상 학습 분포 대비 편차')
    st.dataframe(candidates, hide_index=True, width='stretch')
    if not candidates.empty:
        st.bar_chart(candidates.set_index('변수')['중앙값 대비 IQR 편차'])
        st.caption('변수명의 번호는 관련 측정 스탠드입니다. 참고 구간 이탈 항목부터 운전 조건·센서 값과 정상 참고값을 대조하세요.')
    st.subheader('개별 예측 민감도')
    st.caption('각 변수를 정상 학습 중앙값으로 하나씩 바꿨을 때의 원래 점수 − 변경 점수입니다. 양수는 교체 후 점수가 감소했음을 뜻합니다. 변수 간 상관관계 때문에 비현실적인 조합이 생길 수 있으며 인과적 기여도나 합산 설명으로 해석할 수 없습니다.')
    if st.button('이 사례의 민감도 계산'):
        sensitivity = local_sensitivity(bundle, sample)
        strongest = sensitivity.loc[sensitivity['점수 변화'].abs().nlargest(12).index]
        st.bar_chart(strongest.set_index('변수'))
        st.dataframe(strongest, hide_index=True)
    if st.button('이 사례 점검 기록 저장'):
        score = predict_row(bundle, pd.DataFrame([sample]))
        record_detection(str(uuid.uuid4()), source, model, sample_id, None, score, threshold, sample, bundle)
        st.success('점검 기록을 저장했습니다.')


def new_replay(context, start=0):
    return dict(context=context, position=int(start), run=str(uuid.uuid4()), running=False, processed=0)


def replay_step(state, frame, bundle, threshold, source, model):
    if state['position'] >= len(frame):
        state['running'] = False
        return None
    position = state['position']
    row = frame.iloc[position]
    score = predict_row(bundle, pd.DataFrame([row]))
    step = position + 1
    level=risk_level(score,bundle.get('risk_policy'))
    if score >= threshold or level=='경고':
        record_detection(f"{state['run']}:{position}", source, model, position, step,
                         score, threshold, row, bundle)
    state.update(position=position + 1, processed=state['processed'] + 1,
                 last_score=score, last_step=step, last_label=int(row.label),last_risk=level)
    state.setdefault('risk_trace',[]).append(dict(step=step,score=score,risk_level=level))
    state['risk_trace']=state['risk_trace'][-200:]
    if state['position'] == len(frame):
        state['running'] = False
    return score


def render_replay(frame, bundle, threshold, model):
    st.subheader('기록 데이터 운전 재생')
    st.caption('CAD-TCM 원본 행 순서 재생 · 시뮬레이션 단계는 샘플 순서입니다. 실제 측정 시각·PLC 연결은 제공되지 않습니다.')
    context = json.dumps([model, threshold, len(frame)])
    if 'replay' not in st.session_state or st.session_state.replay['context'] != context:
        st.session_state.replay = new_replay(context)
    state = st.session_state.replay
    speed = st.select_slider('재생 속도 (초당 목표 단계)', options=[1, 2, 4], value=1)
    start = st.number_input('시작 샘플 순번', min_value=0, max_value=len(frame)-1, value=0,
                            disabled=state['running'])
    cols = st.columns(3)
    if cols[0].button('시작 / 계속', disabled=state['position'] >= len(frame)):
        if state['processed'] == 0:
            state['position'] = int(start)
        state['running'] = True
    if cols[1].button('일시정지'):
        state['running'] = False
    if cols[2].button('정지'):
        st.session_state.replay = new_replay(context)
        st.rerun()

    @st.fragment(run_every=1 / speed)
    def tick():
        state = st.session_state.replay
        if state['running']:
            replay_step(state, frame, bundle, threshold, 'CAD-TCM 원본 재생', model)
        st.write('재생 중' if state['running'] else '정지 / 일시정지')
        st.progress(state['position'] / len(frame))
        st.caption(f"처리 {state['processed']:,}건 / 다음 샘플 {state['position']:,}번")
        if 'last_score' in state:
            score = state['last_score']
            st.metric(f"시뮬레이션 단계 {state['last_step']:,} · AI 이상 점수", f'{score:.2%}')
            st.caption(f"실제 라벨 {state['last_label']}")
            st.metric('연구용 위험도',state['last_risk'])
            st.line_chart(pd.DataFrame(state['risk_trace']).set_index('step')['score'])
            st.dataframe(pd.DataFrame(state['risk_trace']),hide_index=True)
            if score >= threshold:
                st.error('이상 감지 · 해당 사례를 이력에 저장했습니다.')
            else:
                st.success('AI 정상 판정')
            if state['last_risk']=='경고' and score<threshold:
                st.warning('위험도 경고 · 이진 판정과 별도 구간이며 점검 이력에 저장했습니다.')
        st.caption('모델 또는 임계값을 변경하면 재생 위치가 초기화됩니다. 저장 이력은 이상 이력 화면에서 확인하세요.')
    tick()
    st.button('시뮬레이션 점검 이력 열기',on_click=lambda:st.session_state.update(page_choice='이상 이력'))
