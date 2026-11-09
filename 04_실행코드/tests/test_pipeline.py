import unittest
import numpy as np
import pandas as pd
from scripts.make_demo_data import make_demo
from src.config import FEATURES
from src.modeling import train_rf, predict_row, split_supervised
from src.insights import get_inspection_candidates
from src.data import validate_and_prepare, DataValidationError

class TestPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df = make_demo(n=350, seed=10)
        cls.bundle, cls.report, cls.eval = train_rf(cls.df, source_kind='synthetic_demo')

    def test_schema(self):
        self.assertEqual(len(FEATURES), 51)
        self.assertEqual(len(set(FEATURES)), 51)
        self.assertEqual(len(self.bundle['features']), 51)

    def test_predict(self):
        prob = predict_row(self.bundle, self.df.iloc[[0]])
        self.assertGreaterEqual(prob, 0.)
        self.assertLessEqual(prob, 1.)
        self.assertIn(self.report['source_kind'], ['synthetic_demo'])
        self.assertGreater(self.report['test_count'], 0)

    def test_no_meta_feature_leak(self):
        self.assertNotIn('label', self.bundle['features'])
        self.assertNotIn('task_id', self.bundle['features'])
        self.assertNotIn('task_split', self.bundle['features'])

    def test_inspection(self):
        checklist = get_inspection_candidates(self.bundle, self.df.iloc[0])
        self.assertIn('정상 참고범위 벗어남', checklist.columns)
        self.assertLessEqual(len(checklist), 8)

    def test_normal_only_official_train_falls_back(self):
        df = self.df.copy()
        df['task_split'] = np.where(df['label'] == 0, 'train', 'test')
        train, test, note = split_supervised(df)
        self.assertEqual(train['label'].nunique(), 2)
        self.assertEqual(test['label'].nunique(), 2)
        self.assertIn('층화', note)

    def test_missing(self):
        with self.assertRaises(DataValidationError):
            validate_and_prepare(pd.DataFrame({'label':[0,1]}))

if __name__ == '__main__':
    unittest.main()
