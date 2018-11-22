from unittest import TestCase
from collections import Counter

import pandas as pd

from scripts.utils import evaluate_f1_initialize_confusion_matrix, add_tags_and_extract_rules
import scripts.vars as my_vars


class TestEvaluateF1InitializeConfusionMatrix(TestCase):
    def test_evaluate_f1_initialize_confusion_matrix(self):
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
        classes = ["apple", "banana"]
        rules = [
            pd.Series({"A": "low", "B": (1, 1), "C": (3, 3), "Class": "apple"}, name=0),
            pd.Series({"A": "low", "B": (1, 1), "C": (2, 2), "Class": "apple"}, name=1),
            pd.Series({"A": "high", "B": (4, 4), "C": (1, 1), "Class": "banana"}, name=2),
            pd.Series({"A": "low", "B": (1.5, 1.5), "C": (0.5, 0.5), "Class": "banana"}, name=3),
            pd.Series({"A": "low", "B": (0.5, 0.5), "C": (3, 3), "Class": "banana"}, name=4),
            pd.Series({"A": "high", "B": (0.75, 0.75), "C": (2, 2), "Class": "banana"}, name=5)
        ]
        min_max = pd.DataFrame({"B": {"min": 1, "max": 5}, "C": {"min": 1, "max": 11}})
        tagged, initial_rules = add_tags_and_extract_rules(df, 2, class_col_name, lookup, min_max, classes)
        print("init rules")
        print(initial_rules)
        print("dataset")
        print(df)
        # for _, e in extracted.iterrows():
        #     print(e)
        #     print(e.name)
        print(my_vars.seed_mapping)
        evaluate_f1_initialize_confusion_matrix(df, initial_rules, class_col_name, lookup, min_max, classes)
        self.fail()
