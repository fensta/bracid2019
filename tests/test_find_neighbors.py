from unittest import TestCase
from collections import Counter

import pandas as pd

from scripts.vars import CONDITIONAL
from scripts.utils import find_nearest_examples


class TestFindNeighbors(TestCase):
    """Test find_neighbors() from utils.py"""

    def test_find_neighbors_too_few(self):
        """Test that warning is thrown if too few neighbors exist"""
        dataset = pd.DataFrame({"A": [1, 2], "B": [1, 2], "C": [2, 2], "D": ["x", "y"], "Class": ["A", "B"]})
        rule = pd.Series({"A": (0.1, 1), "B": (1, 1), "C": (2, 2), "D": "x", "Class": "A"})
        k = 3
        class_col_name = "Class"
        min_max = pd.DataFrame({"A": {"min": 1, "max": 5}, "B": {"min": 1, "max": 11}, "C": {"min": 1, "max": 2}})
        lookup = \
            {
                "D":
                    {
                        'x': 1,
                        'y': 1,
                        CONDITIONAL:
                            {
                                'x':
                                    Counter({
                                        'A': 1
                                    }),
                                'y':
                                    Counter({
                                        'B': 1
                                    })
                            }
                    }
            }
        self.assertWarns(UserWarning, find_nearest_examples, dataset, k, rule, class_col_name, lookup, min_max, None)

    def test_find_neighbors_numeric_nominal(self):
        """Tests what happens if input has a numeric and a nominal feature"""
        df = pd.DataFrame({"A": ["low", "low", "high", "low", "low", "high"], "B": [1, 1, 4, 1.5, 0.5, 0.75],
                           "C": [3, 2, 1, .5, 3, 2],
                           "Class": ["apple", "apple", "banana", "banana", "banana", "banana"]})
        class_col_name = "Class"
        lookup = \
            {
                "A":
                    {
                        'high': 2,
                        'low': 4,
                        CONDITIONAL:
                            {
                                'high':
                                    Counter({
                                        'banana': 2
                                    }),
                                'low':
                                    Counter({
                                        'banana': 2,
                                        'apple': 2
                                    })
                            }
                    }
            }
        k = 4
        correct = None
        if k == 1:
            correct = df.iloc[[5]]
        elif k == 2:
            correct = df.iloc[[5, 2]]
        elif k == 3:
            correct = df.iloc[[5, 2, 3]]
        elif k >= 4:
            correct = df.iloc[[5, 2, 3, 4]]
        rule = pd.Series({"A": "high", "B": (1, 1), "Class": "banana"})
        classes = ["apple", "banana"]
        min_max = pd.DataFrame({"A": {"min": 1, "max": 5}, "B": {"min": 1, "max": 11}})

        neighbors = find_nearest_examples(df, k, rule, class_col_name, lookup, min_max, classes)
        self.assertTrue(neighbors.shape[0] == k)
        self.assertTrue(neighbors.equals(correct))
