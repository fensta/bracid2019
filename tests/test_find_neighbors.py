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
