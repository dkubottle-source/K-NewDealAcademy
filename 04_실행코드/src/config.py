"""CAD-TCM feature names and display labels (not plant-calibrated units)."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / 'data'
MODEL_DIR = ROOT / 'models'
REMOTE_CSV = 'https://huggingface.co/datasets/lifelonglab/CAD-TCM/resolve/main/data.csv'
DATASET_URL = 'https://huggingface.co/datasets/lifelonglab/CAD-TCM'
SEED = 42

FEATURES = (
    ['thickness_entry', 'thickness_exit', 'width', 'ys_entry', 'ys_exit']
    + [f'work_roll_diam_{i}' for i in range(1, 6)]
    + [f'work_roll_mileage_{i}' for i in range(1, 6)]
    + [f'reduction_{i}' for i in range(1, 6)]
    + [f'tension_{i}' for i in range(0, 6)]
    + [f'roll_speed_{i}' for i in range(1, 6)]
    + [f'force_{i}' for i in range(1, 6)]
    + [f'torque_{i}' for i in range(1, 6)]
    + [f'gap_{i}' for i in range(1, 6)]
    + [f'motor_power_{i}' for i in range(1, 6)]
)
META_COLUMNS = {'label', 'task_split', 'task_id', 'task_name'}
GROUPS = {
    '압연력': 'force', '롤 토크': 'torque', '장력': 'tension',
    '롤 속도': 'roll_speed', '압하율': 'reduction',
    '롤 간격': 'gap', '모터 출력': 'motor_power',
}
BASE_LABELS = {
    'thickness_entry':'입측 두께', 'thickness_exit':'출측 두께',
    'width':'소재 폭', 'ys_entry':'입측 항복강도', 'ys_exit':'출측 항복강도',
    'work_roll_diam':'워크롤 직경', 'work_roll_mileage':'워크롤 사용량',
    'reduction':'압하율', 'tension':'장력', 'roll_speed':'롤 속도',
    'force':'압연력', 'torque':'롤 토크', 'gap':'롤 간격', 'motor_power':'모터 출력',
}

def korean_feature(feature: str) -> str:
    if feature in BASE_LABELS:
        return BASE_LABELS[feature]
    for prefix in sorted(BASE_LABELS, key=len, reverse=True):
        if feature.startswith(prefix + '_'):
            suffix = feature[len(prefix)+1:]
            if suffix.isdigit():
                if prefix == 'tension' and suffix == '0':
                    return '입측 장력(0번 위치)'
                return f'{int(suffix)}번 {BASE_LABELS[prefix]}'
    return feature
