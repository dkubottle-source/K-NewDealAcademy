# 프로그램 실행 및 학습

Python 3.11이 필요합니다. 다음 명령은 04_실행코드 폴더에서 실행합니다. 최초 패키지 설치와 CAD-TCM 다운로드에는 인터넷이 필요합니다.

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-lock.txt
.\.venv\Scripts\python.exe -m scripts.make_demo_data
.\.venv\Scripts\python.exe -m scripts.train --data data/synthetic_demo.csv --kind synthetic_demo
.\.venv\Scripts\python.exe -m scripts.migrate_history
.\.venv\Scripts\python.exe -m streamlit run app.py
```

http://127.0.0.1:8501 에 접속합니다. 모델은 포함하지 않았으므로 먼저 학습해야 합니다.

## CAD-TCM 및 4차 비교

가상환경 준비 후 다음 순서로 실행합니다.

```powershell
.\.venv\Scripts\python.exe -m scripts.download_data
.\.venv\Scripts\python.exe -m scripts.train --data data/cad_tcm.csv --kind cad_tcm
.\.venv\Scripts\python.exe -m scripts.tune
.\.venv\Scripts\python.exe -m scripts.experiment_phase4
.\.venv\Scripts\python.exe -m scripts.analyze_phase4
```

기본 RF와 튜닝 후보가 4차 실험의 입력이므로 순서를 지켜야 합니다. 데이터·모델·평가 파일은 실행 시 생성됩니다. 재실행하면 생성된 결과를 덮어쓸 수 있습니다.

## 핵심 테스트

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -p test_pipeline.py -v
```

기존 산출물에 의존하는 전체 회귀 테스트는 이 경량 묶음에서 제외했습니다. 새 환경의 전체 설치·학습을 이번 패키징에서 재실행하지 않았습니다.
