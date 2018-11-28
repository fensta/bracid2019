from unittest import TestCase
from collections import Counter
import copy

import pandas as pd

from scripts.utils import add_one_best_rule, find_nearest_examples
import scripts.vars as my_vars


class TestAddOneBestRule(TestCase):
    """Tests add_one_best_rule() from utils.py"""

    def test_add_one_best_rule_update(self):
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
        # Actually, correctly it should've been
        # my_vars.conf_matrix = {my_vars.TP: {0, 1}, my_vars.FP: set(), my_vars.TN: {2, 5}, my_vars.FN: {3, 4}}
        # at the start (i.e. F1=0.66666), but to see if it changes, it's changed
        my_vars.conf_matrix = {my_vars.TP: {0}, my_vars.FP: set(), my_vars.TN: {1, 2, 5}, my_vars.FN: {3, 4}}
        initial_f1 = 0.1
        k = 3
        neighbors, dists = find_nearest_examples(df, k, rules[0], class_col_name, lookup, min_max, classes,
                                                 use_same_label=True, only_uncovered_neighbors=True)
        improved, updated_rules = add_one_best_rule(df, neighbors, rules[0], rules, initial_f1, class_col_name, lookup,
                                                    min_max, classes)
        correct_closest_rule_per_example = {
            0: (1, 0.010000000000000002),
            1: (0, 0.0),
            2: (5, 0.67015625),
            3: (1, 0.038125),
            4: (0, 0.015625),
            5: (2, 0.67015625)}
        self.assertTrue(improved is True)
        correct_generalized_rule = pd.Series({"A": "low", "B": (1, 1), "C": (2.0, 3), "Class": "apple"}, name=0)
        correct_confusion_matrix = {my_vars.TP: {0, 1}, my_vars.FP: set(), my_vars.TN: {2, 5}, my_vars.FN: {3, 4}}
        # Make sure confusion matrix, closest rule per example, and rule set were updated with the updated rule too
        for example_id in my_vars.closest_rule_per_example:
            rule_id, dist = my_vars.closest_rule_per_example[example_id]
            self.assertTrue(rule_id == correct_closest_rule_per_example[example_id][0] and
                            abs(dist - correct_closest_rule_per_example[example_id][1]) < 0.001)
        self.assertTrue(rules[0].equals(correct_generalized_rule))
        self.assertTrue(my_vars.conf_matrix == correct_confusion_matrix)

    def test_add_one_best_rule_no_update(self):
        """Tests that rule set is not updated when no generalized rule improves F1"""
        # df = pd.DataFrame({"A": ["low", "low", "high", "low", "low", "high"], "B": [1, 1, 4, 1.5, 0.5, 0.75],
        #                    "C": [3, 2, 1, .5, 3, 2],
        #                    "Class": ["apple", "apple", "banana", "banana", "banana", "banana"]})
        # class_col_name = "Class"
        # lookup = \
        #     {
        #         "A":
        #             {
        #                 'high': 2,
        #                 'low': 4,
        #                 my_vars.CONDITIONAL:
        #                     {
        #                         'high':
        #                             Counter({
        #                                 'banana': 2
        #                             }),
        #                         'low':
        #                             Counter({
        #                                 'banana': 2,
        #                                 'apple': 2
        #                             })
        #                     }
        #             }
        #     }
        # classes = ["apple", "banana"]
        # min_max = pd.DataFrame({"B": {"min": 1, "max": 5}, "C": {"min": 1, "max": 11}})
        # my_vars.positive_class = "apple"
        # rules = [
        #     pd.Series({"A": "low", "B": (1, 1), "C": (3, 3), "Class": "apple"}, name=0),
        #     pd.Series({"A": "low", "B": (1, 1), "C": (2, 2), "Class": "apple"}, name=1),
        #     pd.Series({"A": "high", "B": (4, 4), "C": (1, 1), "Class": "banana"}, name=2),
        #     pd.Series({"A": "low", "B": (1.5, 1.5), "C": (0.5, 0.5), "Class": "banana"}, name=3),
        #     pd.Series({"A": "low", "B": (0.5, 0.5), "C": (3, 3), "Class": "banana"}, name=4),
        #     pd.Series({"A": "high", "B": (0.75, 0.75), "C": (2, 2), "Class": "banana"}, name=5)
        # ]
        # my_vars.closest_rule_per_example = {
        #     0: (1, 0.010000000000000002),
        #     1: (0, 0.010000000000000002),
        #     2: (5, 0.67015625),
        #     3: (1, 0.038125),
        #     4: (0, 0.015625),
        #     5: (2, 0.67015625)}
        # correct_closest = copy.deepcopy(my_vars.closest_rule_per_example)
        # my_vars.conf_matrix = {my_vars.TP: {0, 1}, my_vars.FP: set(), my_vars.TN: {2, 5}, my_vars.FN: {3, 4}}
        # correct_f1 = 0.8
        # initial_f1 = 0.1
        # improved, rules = add_one_best_rule(df, rules[0], rules, initial_f1, class_col_name, lookup, min_max, classes)
        # # print(my_vars.closest_rule_per_example)
        # # print(my_vars.conf_matrix)
        # # print(f1)
        # correct_closest_rule_per_example = {
        #     0: (1, 0.010000000000000002),
        #     1: (0, 0.010000000000000002),
        #     2: (5, 0.67015625),
        #     3: (1, 0.038125),
        #     4: (0, 0.0),
        #     5: (2, 0.67015625)}
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
        # Actually, correctly it should've been
        # my_vars.conf_matrix = {my_vars.TP: {0, 1}, my_vars.FP: set(), my_vars.TN: {2, 5}, my_vars.FN: {3, 4}}
        # at the start (i.e. F1=0.66666), but to see if it changes, it's changed
        my_vars.conf_matrix = {my_vars.TP: {0, 1}, my_vars.FP: set(), my_vars.TN: {2, 5}, my_vars.FN: {3, 4}}
        initial_f1 = 0.8
        k = 3
        neighbors, dists = find_nearest_examples(df, k, rules[0], class_col_name, lookup, min_max, classes,
                                                 use_same_label=True, only_uncovered_neighbors=True)
        improved, updated_rules = add_one_best_rule(df, neighbors, rules[0], rules, initial_f1, class_col_name, lookup,
                                                    min_max, classes)
        correct_closest_rule_per_example = {
            0: (1, 0.010000000000000002),
            1: (0, 0.010000000000000002),
            2: (5, 0.67015625),
            3: (1, 0.038125),
            4: (0, 0.015625),
            5: (2, 0.67015625)}
        self.assertTrue(improved is False)
        correct_generalized_rule = pd.Series({"A": "low", "B": (1, 1), "C": (2.0, 3), "Class": "apple"}, name=0)
        correct_confusion_matrix = {my_vars.TP: {0, 1}, my_vars.FP: set(), my_vars.TN: {2, 5}, my_vars.FN: {3, 4}}
        # Make sure confusion matrix, closest rule per example, and rule set were updated with the updated rule too
        for example_id in my_vars.closest_rule_per_example:
            rule_id, dist = my_vars.closest_rule_per_example[example_id]
            self.assertTrue(rule_id == correct_closest_rule_per_example[example_id][0] and
                            abs(dist - correct_closest_rule_per_example[example_id][1]) < 0.001)
        self.assertTrue(rules[0].equals(correct_generalized_rule))
        self.assertTrue(my_vars.conf_matrix == correct_confusion_matrix)
