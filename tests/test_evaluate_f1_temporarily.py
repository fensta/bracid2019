from unittest import TestCase
from collections import Counter
import copy

import pandas as pd

from scripts.utils import evaluate_f1_temporarily
import scripts.vars as my_vars


class TestEvaluateF1Temporarily(TestCase):
    """Tests evaluate_f1_temporarily() in utils.py"""

    def test_evaluate_f1_temporarily(self):
        """Tests that the global variables won't be updated despite local changes"""
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
        my_vars.closest_rule_per_example = {
            0: (1, 0.010000000000000002),
            1: (0, 0.010000000000000002),
            2: (5, 0.67015625),
            3: (1, 0.038125),
            4: (0, 0.015625),
            5: (2, 0.67015625)}
        correct_closest = copy.deepcopy(my_vars.closest_rule_per_example)
        my_vars.conf_matrix = {my_vars.TP: {0, 1}, my_vars.FP: {3, 4}, my_vars.TN: {2, 5}, my_vars.FN: set()}
        new_rule = pd.Series({"A": "low", "B": (0.5, 1.0), "C": (3, 3), "Class": "banana"}, name=0)
        correct_f1 = 0.8

        f1, conf_matrix, closest = evaluate_f1_temporarily(df, new_rule, class_col_name, lookup, min_max, classes)
        correct_closest_rule_per_example = {
            0: (1, 0.010000000000000002),
            1: (0, 0.010000000000000002),
            2: (5, 0.67015625),
            3: (1, 0.038125),
            4: (0, 0.0),
            5: (2, 0.67015625)}
        self.assertTrue(f1 == correct_f1)
        # Local result is still the same as in test_evaluate_f1_update_confusion_matrix.py
        for example_id in closest:
            rule_id, dist = closest[example_id]
            self.assertTrue(rule_id == correct_closest_rule_per_example[example_id][0] and
                            abs(dist - correct_closest_rule_per_example[example_id][1]) < 0.001)
        correct_conf_matrix = {my_vars.TP: {0, 1}, my_vars.FP: {3}, my_vars.TN: {2, 4, 5}, my_vars.FN: set()}
        self.assertTrue(conf_matrix == correct_conf_matrix)
        # But now check that global variables remained unaffected by the changes
        correct_conf_matrix = {my_vars.TP: {0, 1}, my_vars.FP: {3, 4}, my_vars.TN: {2, 5}, my_vars.FN: set()}
        self.assertTrue(my_vars.conf_matrix == correct_conf_matrix)
        self.assertTrue(correct_closest == my_vars.closest_rule_per_example)
