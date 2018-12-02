from unittest import TestCase
from collections import Counter

import pandas as pd

from scripts.utils import evaluate_f1_update_confusion_matrix
import scripts.vars as my_vars


class TestEvaluateF1UpdateConfusionMatrix(TestCase):
    """Tests evaluate_f1_update_confusion_matrix() in utils.py"""

    def test_evaluate_f1_update_confusion_matrix_updated(self):
        """Tests what happens if input has a numeric and a nominal feature and a rule that predicts an example is
        updated"""
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
        my_vars.conf_matrix = {my_vars.TP: {0, 1}, my_vars.FP: {3, 4}, my_vars.TN: {2, 5}, my_vars.FN: set()}
        new_rule = pd.Series({"A": "low", "B": (0.5, 1.0), "C": (3, 3), "Class": "banana"}, name=0)
        # tagged, initial_rules = add_tags_and_extract_rules(df, 2, class_col_name, lookup, min_max, classes)
        correct_f1 = 0.8
        f1 = evaluate_f1_update_confusion_matrix(df, new_rule, class_col_name, lookup, min_max, classes)
        correct_closest_rule_per_example = {
            0: (1, 0.010000000000000002),
            1: (0, 0.010000000000000002),
            2: (5, 0.67015625),
            3: (1, 0.038125),
            4: (0, 0.0),
            5: (2, 0.67015625)}
        self.assertTrue(f1 == correct_f1)
        for example_id in my_vars.closest_rule_per_example:
            rule_id, dist = my_vars.closest_rule_per_example[example_id]
            self.assertTrue(rule_id == correct_closest_rule_per_example[example_id][0] and
                            abs(dist - correct_closest_rule_per_example[example_id][1]) < 0.001)
        correct_conf_matrix = {'tp': {0, 1}, 'fp': {3}, 'tn': {2, 4, 5}, 'fn': set()}
        self.assertTrue(my_vars.conf_matrix == correct_conf_matrix)

    def test_evaluate_f1_update_confusion_matrix_not_updated(self):
        """Tests what happens if input has a numeric and a nominal feature and a rule that predicts an example is
        not updated as F1 score doesn't improve"""
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
        my_vars.conf_matrix = {my_vars.TP: {0, 1}, my_vars.FP: set(), my_vars.TN: {2, 5}, my_vars.FN: {3, 4}}
        new_rule = pd.Series({"A": "low", "B": (0.5, 0.5), "C": (3, 3), "Class": "banana"}, name=4)
        correct_f1 = 2*1*0.5/1.5

        f1 = evaluate_f1_update_confusion_matrix(df, new_rule, class_col_name, lookup, min_max, classes)
        correct_closest_rule_per_example = {
            0: (1, 0.010000000000000002),
            1: (0, 0.010000000000000002),
            2: (5, 0.67015625),
            3: (1, 0.038125),
            4: (0, 0.015625),
            5: (2, 0.67015625)}
        self.assertTrue(f1 == correct_f1)
        for example_id in my_vars.closest_rule_per_example:
            rule_id, dist = my_vars.closest_rule_per_example[example_id]
            self.assertTrue(rule_id == correct_closest_rule_per_example[example_id][0] and
                            abs(dist - correct_closest_rule_per_example[example_id][1]) < 0.001)
        correct_conf_matrix = {'tp': {0, 1}, 'fp': set(), 'tn': {2, 5}, 'fn': {3, 4}}
        self.assertTrue(my_vars.conf_matrix == correct_conf_matrix)
