from unittest import TestCase
from collections import Counter

import pandas as pd

from scripts.utils import extend_rule
import scripts.vars as my_vars


class TestExtendRule(TestCase):
    """Tests extend_rule() from utils.py"""

    def test_extend_rule_only_nominal(self):
        """Test that a rule containing only nominal features is extended correctly"""
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
                        my_vars.CONDITIONAL:
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
        classes = ["apple", "banana"]
        min_max = pd.DataFrame({"B": {"min": 1, "max": 5}, "C": {"min": 1, "max": 11}})
        my_vars.positive_class = "apple"
        rules = [
            pd.Series({"A": "low", "B": (1, 1), "C": (3, 3), "Class": "apple"}, name=0),
            pd.Series({"A": "low", "B": (1, 1), "C": (2, 2), "Class": "apple"}, name=1),
            pd.Series({"A": "high", "B": (4, 4), "C": (1, 1), "Class": "banana"}, name=2),
            pd.Series({"A": "low", "B": (1.5, 1.5), "C": (0.5, 0.5), "Class": "banana"}, name=3),
            pd.Series({"A": "low", "B": (0.5, 0.5), "C": (3, 3), "Class": "banana"}, name=4),
            pd.Series({"A": "high", "B": (0.75, 0.75), "C": (2, 2), "Class": "banana"}, name=5)
        ]
        my_vars.closest_rule_per_example = {
            0: (1, 0.010000000000000002),
            1: (0, 0.010000000000000002),
            2: (5, 0.67015625),
            3: (1, 0.038125),
            4: (0, 0.015625),
            5: (2, 0.67015625)}
        k = 3
        extended_rule = extend_rule(df, k, rules[0], class_col_name, lookup, min_max, classes)
        print("extended rule")
        print(extended_rule)
        self.fail()

    def test_extend_rule_only_numeric(self):
        """Test that a rule containing only numeric features is extended correctly"""
        self.fail()

    def test_extend_rule_only_numeric(self):
        """Test that a rule containing nominal and numeric features is extended correctly"""
        self.fail()
