from unittest import TestCase
from collections import Counter

import pandas as pd

from scripts.utils import svdm, CONDITIONAL


class TestSvdm(TestCase):
    def test_svdm_single_feature(self):
        """Tests that correct svdm is computed for 1 nominal feature"""
        df = pd.DataFrame({"A": ["high", "low", "high", "low", "low", "high"], "B": [3, 2, 1, 1, 1, 2],
                           "Class": ["apple", "apple", "banana", "banana", "banana", "banana"]})
        lookup = {}
        class_idx = 2
        lookup = \
            {
                0:
                    {
                        'high': 3,
                        'low': 3,
                        CONDITIONAL:
                            {
                                'high':
                                    Counter({
                                        'banana': 2,
                                        'apple': 1
                                    }),
                                'low':
                                    Counter({
                                        'banana': 2,
                                        'apple': 1
                                    })
                            }
                    }
            }
        for i, _ in enumerate(df):
            if i == class_idx:
                continue
            f1 = df.iloc[:, i]
            svdm()
        self.fail()