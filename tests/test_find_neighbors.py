from unittest import TestCase
from collections import Counter

import pandas as pd

import scripts.vars as my_vars
from scripts.utils import find_nearest_examples


class TestFindNeighbors(TestCase):
    """Test find_neighbors() from utils.py"""

    def test_find_neighbors_too_few(self):
        """Test that warning is thrown if too few neighbors exist"""
        dataset = pd.DataFrame({"A": [1, 2], "B": [1, 2], "C": [2, 2], "D": ["x", "y"], "Class": ["A", "B"]})
        rule = pd.Series({"A": (0.1, 1), "B": (1, 1), "C": (2, 2), "D": "x", "Class": "A"})
        k = 3
        classes = ["apple", "banana"]
        class_col_name = "Class"
        min_max = pd.DataFrame({"A": {"min": 1, "max": 5}, "B": {"min": 1, "max": 11}, "C": {"min": 1, "max": 2}})
        lookup = \
            {
                "D":
                    {
                        'x': 1,
                        'y': 1,
                        my_vars.CONDITIONAL:
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
        self.assertWarns(UserWarning, find_nearest_examples, dataset, k, rule, class_col_name, lookup, min_max, classes,
                         label_type=my_vars.SAME_LABEL_AS_RULE, only_uncovered_neighbors=False)

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

        neighbors, _ = find_nearest_examples(df, k, rule, class_col_name, lookup, min_max, classes,
                                             label_type=my_vars.SAME_LABEL_AS_RULE, only_uncovered_neighbors=False)
        if neighbors is not None:
            self.assertTrue(neighbors.shape[0] == k)
        print(neighbors)
        print(correct)
        self.assertTrue(neighbors.equals(correct))

    def test_find_neighbors_numeric_nominal_label_type(self):
        """Tests what happens if input has a numeric and a nominal feature and we vary label_type as parameter"""
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
        k = 3
        correct_all = df.iloc[[5, 2, 0]]
        correct_same = df.iloc[[5, 2, 3]]
        correct_opposite = df.iloc[[0, 1]]
        rule = pd.Series({"A": "high", "B": (1, 1), "Class": "banana"})
        classes = ["apple", "banana"]
        min_max = pd.DataFrame({"A": {"min": 1, "max": 5}, "B": {"min": 1, "max": 11}})
        neighbors_all, _ = find_nearest_examples(df, k, rule, class_col_name, lookup, min_max, classes,
                                                 label_type=my_vars.ALL_LABELS, only_uncovered_neighbors=False)
        neighbors_same, _ = find_nearest_examples(df, k, rule, class_col_name, lookup, min_max, classes,
                                                  label_type=my_vars.SAME_LABEL_AS_RULE, only_uncovered_neighbors=False)
        neighbors_opposite, _ = find_nearest_examples(df, k, rule, class_col_name, lookup, min_max, classes,
                                                      label_type=my_vars.OPPOSITE_LABEL_TO_RULE,
                                                      only_uncovered_neighbors=False)
        self.assertTrue(neighbors_all.equals(correct_all))
        self.assertTrue(neighbors_same.equals(correct_same))
        self.assertTrue(neighbors_opposite.equals(correct_opposite))

    def test_find_neighbors_numeric_nominal_covered(self):
            """Tests what happens if input has a numeric and a nominal feature and some examples are already covered
            by the rule"""
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
            k = 4
            correct = None
            if k == 1:
                correct = df.iloc[[5]]
            elif k == 2:
                correct = df.iloc[[5, 2]]
            elif k == 3:
                correct = df.iloc[[5, 2, 3]]
            elif k >= 4:
                # correct = df.iloc[[5, 2, 3, 4]]
                # Examples at indices 2 and 4 are already covered by the rule, so don't return them as neighbors
                my_vars.examples_covered_by_rule = {0: {2, 4}}
                correct = df.iloc[[5, 3]]
            rule = pd.Series({"A": "high", "B": (1, 1), "Class": "banana"}, name=0)
            classes = ["apple", "banana"]
            min_max = pd.DataFrame({"A": {"min": 1, "max": 5}, "B": {"min": 1, "max": 11}})

            neighbors, _ = find_nearest_examples(df, k, rule, class_col_name, lookup, min_max, classes,
                                                 label_type=my_vars.SAME_LABEL_AS_RULE, only_uncovered_neighbors=True)
            self.assertTrue(neighbors.equals(correct))

    def test_find_neighbors_numeric_nominal_stats(self):
            """Tests that global statistics are updated accordingly"""
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
            my_vars.closest_rule_per_example = {
                0: (1, 0.010000000000000002),
                1: (0, 0.010000000000000002),
                2: (5, 0.67015625),
                3: (1, 0.038125),
                4: (0, 0.015625),
                5: (2, 0.67015625)}
            my_vars.closest_examples_per_rule = {0: {1, 4}, 1: {0, 3}, 2: {5}, 5: {2}}
            k = 4
            correct = df.iloc[[5, 2, 3, 4]]
            rule = pd.Series({"A": "high", "B": (1, 1), "Class": "banana"}, name=0)
            classes = ["apple", "banana"]
            my_vars.positive_class = "banana"
            min_max = pd.DataFrame({"A": {"min": 1, "max": 5}, "B": {"min": 1, "max": 11}})
            correct_covered = {}
            correct_examples_per_rule = {0: {1, 2, 4, 5}, 1: {0, 3}}
            correct_closest_rule_per_example = {
                0: (1, 0.010000000000000002), 1: (0, 0.010000000000000002), 2: (0, 0.09), 3: (1, 0.038125),
                4: (0, 0.015625), 5: (0, 0.0006250000000000001)}
            neighbors, _ = find_nearest_examples(df, k, rule, class_col_name, lookup, min_max, classes,
                                                 label_type=my_vars.SAME_LABEL_AS_RULE, only_uncovered_neighbors=False)
            self.assertTrue(neighbors.equals(correct))
            self.assertTrue(correct_covered == my_vars.examples_covered_by_rule)
            self.assertTrue(correct_examples_per_rule == my_vars.closest_examples_per_rule)
            for example_id, (rule_id, dist) in correct_closest_rule_per_example.items():
                self.assertTrue(example_id in my_vars.closest_rule_per_example)
                other_id, other_dist = my_vars.closest_rule_per_example[example_id]
                self.assertTrue(rule_id == other_id)
                self.assertTrue(abs(dist-other_dist) < 0.0001)

    def test_find_neighbors_numeric_nominal_covers(self):
            """Tests that the stats for a newly covered rule are updated (dist = 0)"""
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
            k = 4
            correct = None
            if k == 1:
                correct = df.iloc[[5]]
            elif k == 2:
                correct = df.iloc[[5, 2]]
            elif k == 3:
                correct = df.iloc[[5, 2, 3]]
            elif k >= 4:
                # correct = df.iloc[[5, 2, 3, 4]]
                # Examples at indices 2 and 4 are already covered by the rule, so don't return them as neighbors
                my_vars.examples_covered_by_rule = {0: {2, 4}}
                correct = df.iloc[[5, 3]]
            rule = pd.Series({"A": "high", "B": (1, 1), "Class": "banana"}, name=0)
            classes = ["apple", "banana"]
            min_max = pd.DataFrame({"A": {"min": 1, "max": 5}, "B": {"min": 1, "max": 11}})

            neighbors, _ = find_nearest_examples(df, k, rule, class_col_name, lookup, min_max, classes,
                                                 label_type=my_vars.SAME_LABEL_AS_RULE, only_uncovered_neighbors=True)
            self.assertTrue(neighbors.equals(correct))