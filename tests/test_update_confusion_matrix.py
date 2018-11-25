from unittest import TestCase

import pandas as pd

import scripts.vars as my_vars
from scripts.utils import update_confusion_matrix


class TestUpdateConfusionMatrix(TestCase):
    """Tests update_confusion_matrix() from utils.py"""

    def test_update_confusion_matrix(self):
        """Tests that TP, FN, FP, TN are updated correctly"""
        my_vars.conf_matrix = {my_vars.TP: set(), my_vars.FP: set(), my_vars.TN: set(), my_vars.FN: set()}
        my_vars.conf_matrix[my_vars.TP].add(0)
        my_vars.conf_matrix[my_vars.TN].add(1)
        my_vars.conf_matrix[my_vars.FP].add(2)
        positive_class = "apple"
        class_col_name = "Class"
        examples = [
            pd.Series({"A": "low", "B": (1, 1), "C": (3, 3), "Class": "apple"}, name=3),
            pd.Series({"A": "low", "B": (1, 1), "C": (3, 3), "Class": "banana"}, name=4),
            pd.Series({"A": "low", "B": (1, 1), "C": (3, 3), "Class": "apple"}, name=5),
            pd.Series({"A": "low", "B": (1, 1), "C": (3, 3), "Class": "banana"}, name=6),
        ]
        rules = [
            pd.Series({"A": "low", "B": (1, 1), "C": (3, 3), "Class": "apple"}, name=0),
            pd.Series({"A": "low", "B": (1, 1), "C": (2, 2), "Class": "banana"}, name=1),
        ]
        my_vars.conf_matrix = update_confusion_matrix(examples[0], rules[0], positive_class, class_col_name,
                                                      my_vars.conf_matrix)  # TP
        my_vars.conf_matrix = update_confusion_matrix(examples[1], rules[0], positive_class, class_col_name,
                                                      my_vars.conf_matrix)  # FN
        my_vars.conf_matrix = update_confusion_matrix(examples[2], rules[1], positive_class, class_col_name,
                                                      my_vars.conf_matrix)  # TN
        my_vars.conf_matrix = update_confusion_matrix(examples[3], rules[1], positive_class, class_col_name,
                                                      my_vars.conf_matrix)  # FP

        correct = {
            my_vars.TP: {0, 3},
            my_vars.TN: {1, 6},
            my_vars.FN: {4},
            my_vars.FP: {2, 5},
        }
        self.assertTrue(correct == my_vars.conf_matrix)
