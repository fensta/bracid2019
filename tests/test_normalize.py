from unittest import TestCase
import pandas as pd

from scripts.utils import normalize


class TestNormalize(TestCase):
    """Tests normalize() from utils.py"""

    def test_normalize_ints(self):
        """
        Test that normalization of integers works
        """
        df = pd.DataFrame({"A": [1, 2, 3], "B": [3, 2, 1]})
        df = normalize(df)
        assert(df.shape == (3, 2))
        for col_idx, _ in enumerate(df):
            if col_idx == 0:
                self.assertTrue(df.iloc[:, col_idx].equals(pd.Series([0.0, 0.5, 1.0])))
            else:
                self.assertTrue(df.iloc[:, col_idx].equals(pd.Series([1.0, 0.5, 0.0])))

    def test_normalize_floats(self):
        """
        Test that normalization of floats works
        """
        df = pd.DataFrame({"A": [1.4, 2.4, 3.4], "B": [3.4, 2.4, 1.4]})
        df = normalize(df)
        assert(df.shape == (3, 2))
        for col_idx, _ in enumerate(df):
            if col_idx == 0:
                print(df.iloc[:, col_idx].equals(pd.Series([0.0, 0.5, 1.0])))
                self.assertTrue(df.iloc[:, col_idx].equals(pd.Series([0.0, 0.5, 1.0])))
            else:
                self.assertTrue(df.iloc[:, col_idx].equals(pd.Series([1.0, 0.5, 0.0])))

    def test_normalize_nominal(self):
        """
        Test that normalization is applied only to columns with numeric values
        """
        df = pd.DataFrame({"A": [1, 2, 3], "B": [3.4, 2.4, 1.4], "C": ["A", "B", "C"]})
        df = normalize(df)
        assert (df.shape == (3, 3))
        for col_idx, _ in enumerate(df):
            if col_idx == 0:
                print(df.iloc[:, col_idx].equals(pd.Series([0.0, 0.5, 1.0])))
                self.assertTrue(df.iloc[:, col_idx].equals(pd.Series([0.0, 0.5, 1.0])))
            elif col_idx == 1:
                self.assertTrue(df.iloc[:, col_idx].equals(pd.Series([1.0, 0.5, 0.0])))
            else:
                self.assertTrue(df.iloc[:, col_idx].equals(pd.Series(["A", "B", "C"])))
