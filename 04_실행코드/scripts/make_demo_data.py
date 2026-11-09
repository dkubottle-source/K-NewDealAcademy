"""Generate small, clearly SYNTHETIC test fixtures (not CAD-TCM records)."""
from pathlib import Path
import numpy as np
import pandas as pd
from src.config import DATA_DIR, SEED


def make_demo(n=2400, seed=SEED):
    rng = np.random.default_rng(seed)
    records = []
    for k in range(n):
        anomaly = int(rng.random() < .27)
        problem_stand = int(rng.integers(1, 6))
        failure_mode = int(rng.integers(0, 4))
        thickness = float(rng.choice([2.6, 2.9, 3.3, 3.9, 4.0]))
        target = thickness * rng.uniform(.28, .42)
        r = {'thickness_entry':thickness,'thickness_exit':target,
             'width':float(rng.uniform(980, 1300)),
             'ys_entry':float(rng.uniform(270, 480)),
             'ys_exit':float(rng.uniform(490, 710))}
        nominal_force = np.array([12.1, 9.6, 7.7, 5.6, 4.1]) * 1e6 * (thickness/3.2)**.7
        nominal_torque = np.array([155, 113, 91, 68, 53]) * 1e3 * (thickness/3.2)**.7
        nominal_tension = np.array([73, 135, 170, 207, 241, 258]) * 1e3 * (thickness/3.2)**.6
        speed0 = float(rng.uniform(3.8, 7.0))
        for j in range(1, 6):
            r[f'work_roll_diam_{j}'] = float(rng.uniform(540, 590))
            r[f'work_roll_mileage_{j}'] = float(rng.uniform(0, 120))
            r[f'reduction_{j}'] = float([.35,.26,.17,.11,.06][j-1]*rng.uniform(.9,1.1))
            r[f'roll_speed_{j}'] = float(speed0*(1.25 ** (j-1))*rng.uniform(.97,1.03))
            r[f'force_{j}'] = float(nominal_force[j-1]*rng.uniform(.95,1.05))
            r[f'torque_{j}'] = float(nominal_torque[j-1]*rng.uniform(.94,1.06))
            r[f'gap_{j}'] = float([1.9,1.5,1.2,1.05,.91][j-1]*rng.uniform(.9,1.1))
        for j in range(6):
            r[f'tension_{j}'] = float(nominal_tension[j]*rng.uniform(.91,1.09))
        if anomaly:
            if failure_mode == 0:
                r[f'force_{problem_stand}'] *= rng.uniform(1.24,1.58)
                r[f'reduction_{problem_stand}'] *= rng.uniform(1.18,1.55)
            elif failure_mode == 1:
                r[f'torque_{problem_stand}'] *= rng.uniform(1.35,1.9)
            elif failure_mode == 2:
                r[f'tension_{problem_stand}'] *= rng.uniform(1.5,2.0)
            else:
                r[f'roll_speed_{problem_stand}'] *= rng.uniform(.55,.76)
        for j in range(1, 6):
            r[f'motor_power_{j}'] = float(r[f'torque_{j}']*r[f'roll_speed_{j}']*.022)
        r['label'] = anomaly
        r['task_split'] = 'train' if rng.random() < .8 else 'test'
        r['task_id'] = int(rng.integers(0,8))
        r['task_name'] = f"tcm_{r['task_id']}"
        records.append(r)
    return pd.DataFrame.from_records(records)

if __name__ == '__main__':
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    path=DATA_DIR/'synthetic_demo.csv'
    make_demo().to_csv(path, index=False, float_format='%.9g')
    print(f'연습용 합성 데이터 생성: {path} (실제 CAD-TCM 샘플 아님)')
