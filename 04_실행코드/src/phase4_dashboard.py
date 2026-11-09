"""Phase4 comparison and task views read only executed experiment artifacts."""
import pandas as pd
import streamlit as st
from .config import ROOT,korean_feature

OUT=ROOT/'reports/phase4'


@st.cache_data(show_spinner=False)
def artifact_csv(path,modified):
    return pd.read_csv(path,float_precision='round_trip')


def read(name):
    path=OUT/name
    return artifact_csv(str(path),path.stat().st_mtime)


def render_comparison():
    st.subheader('AI 모델 비교')
    if not (OUT/'comparison.csv').exists():
        st.info('4차 실험이 아직 완료되지 않았습니다.');return
    frame=read('comparison.csv')
    protocol=st.selectbox('비교 평가 구간',['nested_selected','nested_fixed_0.5','nested_sigmoid','nested_baseline',
                                      'development_oof_selection','reused_6_7_research','reused_6_7_stored'])
    selected=frame[frame.protocol.eq(protocol)]
    columns=['model','precision','recall','f1','roc_auc','average_precision',
             'false_alarms_per_1000_normal','misses_per_1000_anomalies','worst_task_f1','worst_task_recall','macro_task_ap']
    st.dataframe(selected[columns],hide_index=True,width='stretch')
    st.bar_chart(selected.set_index('model')[['precision','recall','f1','average_precision']],stack=False)
    st.caption('nested는 개발 태스크의 바깥 홀드아웃 평가입니다. development_oof_selection은 최종 선택에 사용한 값으로 낙관적일 수 있습니다. 6~7은 기존에 본 평가 구간이며 새 외부 검증이 아닙니다. pooled AP와 태스크 평균 AP를 함께 확인하세요.')
    if (OUT/'operating_point_comparison.csv').exists():
        st.subheader('경고·주의 cutoff를 이진 기준으로 쓴 비교')
        st.dataframe(read('operating_point_comparison.csv')[['model','operating_point','recall','f1',
            'false_alarms_per_1000_normal','missed_anomalies','worst_task_recall']],hide_index=True)
        st.caption('사전에 정의한 내부 Recall 0.70/0.90 cutoff의 바깥 홀드아웃 결과입니다. 사이드바의 주의 기준 적용 버튼으로 낮은 cutoff를 시험할 수 있습니다. 위험도 정책 자체는 바뀌지 않습니다.')
    st.info('왼쪽 데이터 및 학습 모델에서 4차 연구 후보를 선택하면 모든 기존 예측·모니터링·점검 기능으로 사용할 수 있습니다. 기본 모델은 유지됩니다.')
    st.button('선택 모델로 AI 예측 열기',on_click=lambda:st.session_state.update(page_choice='AI 이상 탐지'))
    st.subheader('확률 보정 비교 (바깥 홀드아웃)')
    st.dataframe(read('calibration_comparison.csv')[['model','fold','calibration','brier','log_loss','average_precision']],hide_index=True)
    st.caption('Brier·Log loss는 작을수록 좋지만 클래스 비율의 영향이 큽니다. 시그모이드 보정도 새로운 운전 조건의 실제 고장 확률을 보장하지 않습니다. 보정 후보는 비교용이며 예측 모델로 채택하지 않았습니다.')
    st.subheader('개발 OOF 임계값 곡선')
    curves=read('threshold_curves.csv.gz')
    model=st.selectbox('임계값 분석 모델',curves.model.unique())
    curve=curves[curves.model.eq(model)]
    # Keep browser payload bounded, include the full CSV download below.
    subset=curve.iloc[::max(1,len(curve)//500)]
    st.line_chart(subset.set_index('threshold')[['recall','f1']])
    st.line_chart(subset.set_index('threshold')['false_alarms_per_1000_normal'])
    st.line_chart(subset.sort_values('recall').set_index('recall')['precision'])
    st.dataframe(curve.iloc[(curve.threshold-.5).abs().argsort().head(1)][['threshold','tn','fp','fn','tp']],hide_index=True)
    st.download_button('전체 임계값·혼동행렬 CSV',curve.to_csv(index=False).encode('utf-8-sig'),'threshold_analysis.csv','text/csv')


def render_task_performance(frame):
    st.subheader('태스크별 성능')
    if not (OUT/'task_metrics.csv').exists():
        st.info('4차 실험이 아직 완료되지 않았습니다.');return
    task=st.selectbox('분석 태스크',list(range(8)),index=3)
    table=read('task_metrics.csv')
    protocol='nested_selected' if task<6 else 'reused_6_7_research'
    rows=table[(table.task_id==task)&(table.protocol==protocol)]
    st.dataframe(rows[['model','normal','anomalies','precision','recall','f1','false_alarms_per_1000_normal',
                      'missed_anomalies','misses_per_1000_anomalies']],hide_index=True)
    st.bar_chart(rows.set_index('model')[['f1','recall']],stack=False)
    st.caption('태스크 3·4·5도 다른 태스크와 동일하게 홀드아웃했습니다. 태스크 6·7 결과는 재평가이며 모델·임계값 선택에 사용하지 않았습니다.')
    if 'task_id' in frame:
        values=frame[frame.task_id.eq(task)]
        st.write(f'현재 데이터에서 선택 태스크 {len(values):,}건')
        if not values.empty:
            if 'label' in values and values.label.notna().any():
                st.bar_chart(values.label.value_counts().rename(index={0:'정상',1:'이상'}))
            else:
                st.info('업로드 데이터에 실제 라벨이 없어 정상·이상 분포는 표시하지 않습니다.')
            feature=st.selectbox('분포 비교 변수',[c for c in ['force_3','torque_3','tension_3','roll_speed_3'] if c in frame],format_func=korean_feature)
            st.dataframe(values.groupby('label')[feature].describe() if 'label' in values else values[feature].describe())
    plot=OUT/'plots/feature_distributions.png'
    if plot.exists():st.image(str(plot),caption='공정 변수의 태스크별 분포 (운전 조건 차이이며 원인 확정이 아님)')
    if task in [3,4,5]:
        st.info('중점 점검 태스크입니다. 분포 차이와 오경보·미탐은 관찰 사실이며 둘 사이의 인과관계는 확인하지 않았습니다.')
