from unittest import TestCase
from collections import Counter
import copy

import pandas as pd

from scripts.utils import evaluate_f1_temporarily
import scripts.vars as my_vars


class TestEvaluateF1Temporarily(TestCase):
    """Tests evaluate_f1_temporarily() in utils.py"""

    def test_evaluate_f1_temporarily(self):
        """Tests that the global confusion matrix won't be updated despite local changes"""
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
        my_vars.conf_matrix = {my_vars.TP: {0, 1}, my_vars.FP: set(), my_vars.TN: {2, 5}, my_vars.FN: {3, 4}}
        new_rule = pd.Series({"A": "low", "B": (0.5, 1.0), "C": (3, 3), "Class": "banana"}, name=0)
        # tagged, initial_rules = add_tags_and_extract_rules(df, 2, class_col_name, lookup, min_max, classes)
        # print("init rules")
        # print(rules)
        # print("dataset")
        # print(df)
        # print(my_vars.seed_mapping)
        correct_f1 = 0.8

        f1, conf_matrix, closest = evaluate_f1_temporarily(df, new_rule, class_col_name, lookup, min_max, classes)
        # print(my_vars.closest_rule_per_example)
        # print(my_vars.conf_matrix)
        # print(f1)
        correct_closest_rule_per_example = {
            0: (1, 0.010000000000000002),
            1: (0, 0.010000000000000002),
            2: (5, 0.67015625),
            3: (1, 0.038125),
            4: (0, 0.0),
            5: (2, 0.67015625)}
        # self.assertTrue(f1 == correct_f1)
        print("deep copy closest rule per example")
        print(correct_closest)
        print("original")
        print(my_vars.closest_rule_per_example)
        print("local")
        print(closest)
        print("local conf matrix")
        print(conf_matrix)
        print("global conf matrix")
        print(my_vars.conf_matrix)
        for example_id in my_vars.closest_rule_per_example:
            rule_id, dist = my_vars.closest_rule_per_example[example_id]
            self.assertTrue(rule_id == correct_closest_rule_per_example[example_id][0] and
                            abs(dist - correct_closest_rule_per_example[example_id][1]) < 0.001)