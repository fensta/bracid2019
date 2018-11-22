from unittest import TestCase
from collections import Counter

import pandas as pd

from scripts.utils import evaluate_f1_initialize_confusion_matrix, extract_initial_rules, add_tags_and_extract_rules
from scripts.vars import CONDITIONAL, seed_mapping


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
                        CONDITIONAL:
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

        tagged, initial_rules = add_tags_and_extract_rules(df, 2, class_col_name, lookup, min_max, classes)
        print("init rules")
        print(initial_rules)
        for r in initial_rules:
            print("rule:\n{}".format(r))
        print("dataset")
        print(df)
        seed_mapping(zip(df.row))
        # for _, e in extracted.iterrows():
        #     print(e)
        #     print(e.name)
        rules = [pd.Series({"A": "high", "B": (1, 1), "Class": "banana"}, name=0),
                 pd.Series({"A": "high", "B": (1, 1), "Class": "banana"}, name=1)]

        evaluate_f1_initialize_confusion_matrix(df, rules, class_col_name, lookup, min_max, classes)
        self.fail()
