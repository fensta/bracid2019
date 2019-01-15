from unittest import TestCase
from collections import Counter

import pandas as pd

from scripts.utils import bracid, Bounds
import scripts.vars as my_vars


class TestBracid(TestCase):
    """Tests bracid() from utils.py"""

    def test_bracid_no_neighbor(self):
        """Tests that rules are discarded if there are no neighbors and the seed belongs to the majority class"""
        df = pd.DataFrame({"A": ["low", "low", "high", "low", "low", "high"], "B": [1, 1, 4, 1.5, 0.5, 0.75],
                           "C": [3, 2, 1, .5, 3, 2],
                           "Class": ["apple", "banana", "banana", "banana", "banana", "banana"]})
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
        # Use majority class as minority to have multiple neighbors and see if the function works correctly
        my_vars.minority_class = "apple"
        rules = [
            pd.Series({"A": "low", "B": Bounds(lower=1, upper=1), "C": Bounds(lower=3, upper=3), "Class": "apple"},
                      name=0),
            pd.Series({"A": "low", "B": Bounds(lower=1, upper=1), "C": Bounds(lower=2, upper=2), "Class": "apple"},
                      name=1),
        ]
        initial_correct_closest_rule_per_example = {
            0: (1, 0.010000000000000002),
            1: (0, 0.010000000000000002),
            2: (5, 0.67015625),
            3: (1, 0.038125),
            4: (0, 0.015625),
            5: (2, 0.67015625)}
        minority_label = "apple"
        k = 3
        rules = bracid(df, k, class_col_name, lookup, min_max, classes, minority_label)
        print("generalized rules")
        print(rules)
        self.fail()

    def test_bracid_stops(self):
        """Tests that the method stops"""
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
        # Use majority class as minority to have multiple neighbors and see if the function works correctly
        my_vars.minority_class = "banana"
        minority_label = "banana"
        k = 3
        correct_rules = [
            pd.Series({"A": "high", "B": Bounds(lower=0.75, upper=4.0), "C": Bounds(lower=1.0, upper=2.0),
                       "Class": "banana"}, name=2),
            pd.Series({"A": "low", "B": (0.5, 1.5), "C": (0.5, 3.0), "Class": "banana"}, name=3),
            pd.Series({"A": "low", "B": Bounds(lower=0.5, upper=1.5), "C": Bounds(lower=0.5, upper=3.0),
                       "Class": "banana"}, name=4),
            pd.Series({"A": "high", "B": Bounds(lower=0.75, upper=4.0), "C": Bounds(lower=1.0, upper=2.0),
                       "Class": "banana"}, name=5),
            pd.Series({"A": "low", "B": Bounds(lower=0.875, upper=2.5), "C": Bounds(lower=1.5, upper=3.0),
                       "Class": "apple"}, name=0),
            pd.Series({"A": "low", "B": Bounds(lower=0.875, upper=2.5), "C": Bounds(lower=1.5, upper=2.0),
                       "Class": "apple"}, name=1),
            pd.Series({"A": "low", "B": Bounds(lower=0.5, upper=1.5), "C": Bounds(lower=0.5, upper=3.0),
                       "Class": "apple"}, name= 6),
            pd.Series({"A": "low", "B": Bounds(lower=0.5, upper=1.5), "C": Bounds(lower=0.5, upper=3.0),
                       "Class": "apple"}, name=7),
            pd.Series({"A": "low", "B": Bounds(lower=0.5, upper=1.5), "C": Bounds(lower=0.5, upper=3.0),
                       "Class": "apple"}, name=9),
            pd.Series({"A": "low", "B": Bounds(lower=0.5, upper=1.5), "C": Bounds(lower=0.5, upper=3.0),
                       "Class": "apple"}, name=10)
        ]
        # Duplicates: 6+7+9+10, 2+5, 3+4
        rules = bracid(df, k, class_col_name, lookup, min_max, classes, minority_label)
        print("generalized rules")
        print(rules)
        print(my_vars.closest_rule_per_example)
        print(my_vars.closest_examples_per_rule)
        print(my_vars.conf_matrix)
        print(my_vars.all_rules)
        print(my_vars.seed_rule_example)
        print(my_vars.seed_example_rule)
        print(my_vars.examples_covered_by_rule)
        self.assertTrue(correct_rules == rules)
