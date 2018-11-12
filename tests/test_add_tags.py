from unittest import TestCase
from collections import Counter

import pandas as pd

from scripts.utils import add_tags, CONDITIONAL, TAG, BORDERLINE, SAFE


class TestAddTags(TestCase):
    """Tests add_tags() from utils.py"""

    def test_add_tags(self):
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
        correct = pd.DataFrame({"A": ["low", "low", "high", "low", "low", "high"], "B": [1, 1, 4, 1.5, 0.5, 0.75],
                                "C": [3, 2, 1, .5, 3, 2],
                                "Class": ["apple", "apple", "banana", "banana", "banana", "banana"],
                                TAG: [BORDERLINE, BORDERLINE, SAFE, BORDERLINE, BORDERLINE, BORDERLINE]
                                })
        classes = ["apple", "banana"]
        min_max = pd.DataFrame({"C": {"min": 1, "max": 5}, "B": {"min": 1, "max": 11}})
        k = 3
        tagged = add_tags(df, k, class_col_name, lookup, min_max, classes)
        # Due to floating point precision, use approximate comparison
        self.assertTrue(tagged.equals(correct))
