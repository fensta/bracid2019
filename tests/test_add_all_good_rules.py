from unittest import TestCase
from collections import Counter

import pandas as pd

from scripts.utils import add_all_good_rules, find_nearest_examples, evaluate_f1_initialize_confusion_matrix
import scripts.vars as my_vars


class TestAddAllGoodRules(TestCase):
    """Tests add_all_good_rules() in utils.py"""

    def test_add_all_good_rules(self):
        """Tests that rule set is updated when a generalized rule improves F1"""
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
        my_vars.positive_class = "banana"
        rules = [
            pd.Series({"A": "low", "B": (1, 1), "C": (3, 3), "Class": "apple"}, name=0),
            pd.Series({"A": "low", "B": (1, 1), "C": (2, 2), "Class": "apple"}, name=1),
            pd.Series({"A": "high", "B": (4, 4), "C": (1, 1), "Class": "banana"}, name=2),
            pd.Series({"A": "low", "B": (1.5, 1.5), "C": (0.5, 0.5), "Class": "banana"}, name=3),
            pd.Series({"A": "low", "B": (0.5, 0.5), "C": (3, 3), "Class": "banana"}, name=4),
            pd.Series({"A": "high", "B": (0.75, 0.75), "C": (2, 2), "Class": "banana"}, name=5)
        ]
        initial_correct_closest_rule_per_example = {
            0: (1, 0.010000000000000002),
            1: (0, 0.010000000000000002),
            2: (5, 0.67015625),
            3: (1, 0.038125),
            4: (0, 0.015625),
            5: (2, 0.67015625)}
        initial_f1 = evaluate_f1_initialize_confusion_matrix(df, rules, class_col_name, lookup, min_max, classes)
        correct_confusion_matrix = {my_vars.TP: {2, 5}, my_vars.FP: set(), my_vars.TN: {0, 1}, my_vars.FN: {3, 4}}
        print(my_vars.conf_matrix)
        self.assertTrue(my_vars.conf_matrix == correct_confusion_matrix)

        # Make sure confusion matrix, closest rule per example are correct at the beginning
        for example_id in my_vars.closest_rule_per_example:
            rule_id, dist = my_vars.closest_rule_per_example[example_id]
            self.assertTrue(rule_id == initial_correct_closest_rule_per_example[example_id][0] and
                            abs(dist - initial_correct_closest_rule_per_example[example_id][1]) < 0.001)

        correct_initial_f1 = 2 * 0.5 * 1 / 1.5
        self.assertTrue(initial_f1 == correct_initial_f1)
        print("initial confusion matrix")
        print(my_vars.conf_matrix)
        print("initial closest rule per example")
        print(my_vars.closest_rule_per_example)
        k = 3
        neighbors, dists = find_nearest_examples(df, k, rules[2], class_col_name, lookup, min_max, classes,
                                                 label_type=my_vars.SAME_LABEL_AS_RULE, only_uncovered_neighbors=True)
        improved, updated_rules = add_all_good_rules(df, neighbors, rules[2], rules, initial_f1, class_col_name, lookup,
                                                     min_max, classes)
        self.assertTrue(improved is True)
        print(my_vars.conf_matrix)
        print(my_vars.examples_covered_by_rule)
        print(my_vars.closest_rule_per_example)
        correct_confusion_matrix = {my_vars.TP: {2, 3, 4, 5}, my_vars.FP: {1}, my_vars.TN: {0}, my_vars.FN: set()}
        correct_closest_rule_per_example = {
            0: (1, 0.010000000000000002),
            1: (2, 0.0),
            2: (5, 0.67015625),
            3: (2, 0.0),
            4: (2, 0.013906250000000002),
            5: (2, 0.0)}
        for example_id in my_vars.closest_rule_per_example:
            rule_id, dist = my_vars.closest_rule_per_example[example_id]
            self.assertTrue(rule_id == correct_closest_rule_per_example[example_id][0] and
                            abs(dist - correct_closest_rule_per_example[example_id][1]) < 0.001)
        self.assertTrue(my_vars.conf_matrix == correct_confusion_matrix)
