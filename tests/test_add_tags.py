from unittest import TestCase
from collections import Counter

import pandas as pd
import numpy as np

from scripts.vars import CONDITIONAL, TAG, BORDERLINE, SAFE, NOISY
from scripts.utils import add_tags


class TestAddTags(TestCase):
    """Tests add_tags() from utils.py"""

    def test_add_tags_safe_borderline(self):
        """Add tags when using nominal and numeric features assigning borderline and safe as tags"""
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
        rules = [
            pd.Series({"A": "low", "B": (1, 1), "C": (3, 3), "Class": "apple"}, name=0),
            pd.Series({"A": "low", "B": (1, 1), "C": (2, 2), "Class": "apple"}, name=1),
            pd.Series({"A": "high", "B": (4, 4), "C": (1, 1), "Class": "banana"}, name=2),
            pd.Series({"A": "low", "B": (1.5, 1.5), "C": (0.5, 0.5), "Class": "banana"}, name=3),
            pd.Series({"A": "low", "B": (0.5, 0.5), "C": (3, 3), "Class": "banana"}, name=4),
            pd.Series({"A": "high", "B": (0.75, 0.75), "C": (2, 2), "Class": "banana"}, name=5)
        ]
        tagged = add_tags(df, k, rules, class_col_name, lookup, min_max, classes)
        # Due to floating point precision, use approximate comparison
        self.assertTrue(tagged.equals(correct))

    def test_add_tags_noisy_safe(self):
        """Add tags when using nominal and numeric features and assigning noisy and safe as tags"""
        df = pd.DataFrame({"A": ["low", "low", "high", "low", "low", "high"], "B": [1, 1, 4, 1.5, 0.5, 0.75],
                           "C": [3, 2, 1, .5, 3, 2],
                           "Class": ["apple", "banana", "banana", "banana", "banana", "banana"]})
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
                                "Class": ["apple", "banana", "banana", "banana", "banana", "banana"],
                                TAG: [NOISY, SAFE, SAFE, SAFE, SAFE, SAFE]
                                })
        classes = ["apple", "banana"]
        min_max = pd.DataFrame({"C": {"min": 1, "max": 5}, "B": {"min": 1, "max": 11}})
        k = 3
        rules = [
            pd.Series({"A": "low", "B": (1, 1), "C": (3, 3), "Class": "apple"}, name=0),
            pd.Series({"A": "low", "B": (1, 1), "C": (2, 2), "Class": "banana"}, name=1),
            pd.Series({"A": "high", "B": (4, 4), "C": (1, 1), "Class": "banana"}, name=2),
            pd.Series({"A": "low", "B": (1.5, 1.5), "C": (0.5, 0.5), "Class": "banana"}, name=3),
            pd.Series({"A": "low", "B": (0.5, 0.5), "C": (3, 3), "Class": "banana"}, name=4),
            pd.Series({"A": "high", "B": (0.75, 0.75), "C": (2, 2), "Class": "banana"}, name=5)
        ]
        tagged = add_tags(df, k, rules, class_col_name, lookup, min_max, classes)
        # Due to floating point precision, use approximate comparison
        self.assertTrue(tagged.equals(correct))

    def test_add_tags_nan(self):
        """Add tags when using nominal and numeric features when all examples contain at least one NaN value"""
        df = pd.DataFrame({"A": [np.NaN, np.NaN, "high", np.NaN, "low", np.NaN],
                           "B": [np.NaN, 1, np.NaN, 1.5, np.NaN, 0.75],
                           "C": [3, 2, 1, .5, 3, 2],
                           "Class": ["apple", "apple", "banana", "banana", "banana", "banana"]})
        class_col_name = "Class"
        lookup = \
            {
                "A":
                    {
                        'high': 1,
                        'low': 1,
                        CONDITIONAL:
                            {
                                'high':
                                    Counter({
                                        'banana': 1
                                    }),
                                'low':
                                    Counter({
                                        'banana': 1
                                    })
                            }
                    }
            }
        correct = pd.DataFrame({"A": [np.NaN, np.NaN, "high", np.NaN, "low", np.NaN],
                                "B": [np.NaN, 1, np.NaN, 1.5, np.NaN, 0.75],
                                "C": [3, 2, 1, .5, 3, 2],
                                "Class": ["apple", "apple", "banana", "banana", "banana", "banana"],
                                TAG: [BORDERLINE, BORDERLINE, SAFE, SAFE, BORDERLINE, BORDERLINE]
                                })
        classes = ["apple", "banana"]
        min_max = pd.DataFrame({"C": {"min": 1, "max": 5}, "B": {"min": 1, "max": 11}})
        rules = [
            pd.Series({"A": np.NaN, "B": (np.NaN, np.NaN), "C": (3, 3), "Class": "apple"}, name=0),
            pd.Series({"A": np.NaN, "B": (1, 1), "C": (2, 2), "Class": "apple"}, name=1),
            pd.Series({"A": "high", "B": (np.NaN, np.NaN), "C": (1, 1), "Class": "banana"}, name=2),
            pd.Series({"A": np.NaN, "B": (1.5, 1.5), "C": (0.5, 0.5), "Class": "banana"}, name=3),
            pd.Series({"A": "low", "B": (np.NaN, np.NaN), "C": (3, 3), "Class": "banana"}, name=4),
            pd.Series({"A": np.NaN, "B": (0.75, 0.75), "C": (2, 2), "Class": "banana"}, name=5)
        ]
        k = 3
        tagged = add_tags(df, k, rules, class_col_name, lookup, min_max, classes)
        # Due to floating point precision, use approximate comparison
        self.assertTrue(tagged.equals(correct))

    def test_add_tags_all_tags(self):
        """Add tags when using nominal and numeric features and assigning noisy, borderline and safe as tags"""
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
                                "Class": ["apple", "banana", "banana", "banana", "banana", "banana"],
                                TAG: [NOISY, SAFE, SAFE, SAFE, SAFE, SAFE]
                                })
        classes = ["apple", "banana"]
        min_max = pd.DataFrame({"C": {"min": 1, "max": 5}, "B": {"min": 1, "max": 11}})
        k = 3
        rules = [
            pd.Series({"A": "low", "B": (1, 1), "C": (3, 3), "Class": "apple"}, name=0),
            pd.Series({"A": "low", "B": (1, 1), "C": (2, 2), "Class": "apple"}, name=1),
            pd.Series({"A": "high", "B": (4, 4), "C": (1, 1), "Class": "banana"}, name=2),
            pd.Series({"A": "low", "B": (1.5, 1.5), "C": (0.5, 0.5), "Class": "banana"}, name=3),
            pd.Series({"A": "low", "B": (0.5, 0.5), "C": (3, 3), "Class": "banana"}, name=4),
            pd.Series({"A": "high", "B": (0.75, 0.75), "C": (2, 2), "Class": "banana"}, name=5)
        ]
        tagged = add_tags(df, k, rules, class_col_name, lookup, min_max, classes)
        print(tagged)
        # Due to floating point precision, use approximate comparison
        self.assertTrue(tagged.equals(correct))
