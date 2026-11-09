## 3. 모델

RandomForest, ExtraTrees, HistGradientBoosting, LogisticRegression 네 알고리즘에서 각각 두 설정을 비교했다. label·task_id·task_name·task_split은 입력 특징에서 제외했다. LogisticRegression에는 학습 폴드의 결측값 중앙값 대체와 표준화를 적용했다.

개발 태스크를 바깥 홀드아웃 [0,3], [1,4], [2,5]로 나누고 각 학습 구간 안에서 GroupKFold 3분할 OOF로 설정과 임계값을 선택했다. 같은 태스크가 해당 학습과 검증에 동시에 포함되지 않도록 구성했다.

내부 Recall 0.70 이상을 만족하는 가장 높은 cutoff를 선택했다. 바깥 테스트 라벨은 이 선택에 사용하지 않았다. LogisticRegression의 최종 개발 OOF cutoff 약 0.4122와 시연용 바깥 폴드의 0.476708873137658은 서로 다른 학습·선택 구간의 값이다.

기본 분석 앱은 기존 RF를 유지하고 연구 후보를 선택할 수 있다. HTML은 저장된 LogisticRegression 점수를 표시하며 브라우저에서 모델을 새로 추론하지 않는다.

## 평가 결과

[모델 성능 비교와 그래프](성능비교.md)

[원본 비교 CSV](근거자료/comparison.csv) · [태스크별 평가](근거자료/task_metrics.csv) · [기존 테스트 기록](근거자료/tests.log)
