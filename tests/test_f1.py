from unittest import TestCase

from scripts.utils import f1


class TestF1(TestCase):
    """Test f1() in utils.py"""

    def test_f1_exception(self):
        """Tests if exception is thrown in case of unequal list lengths"""
        predicted = ["a", "b", "a"]
        true = ["a", "a"]
        positive = "a"
        self.assertRaises(Exception, f1, predicted, true, positive)

    def test_f1_high_recall(self):
        """Tests if F1 is computed correctly"""
        predicted = ["b", "a", "a", "a", "a", "a", "a", "b"]
        true = ["a", "a", "a", "b", "b", "b", "b", "b"]
        positive = "a"
        score = f1(predicted, true, positive)
        correct = 2*1/3*2/3
        self.assertTrue(score == correct)

    def test_f1_high_precision(self):
        """Tests if F1 is computed correctly"""
        predicted = ["a", "a", "b", "b", "a", "b", "a", "b"]
        true = ["a", "a", "a", "a", "b", "a", "a", "a"]
        positive = "a"
        score = f1(predicted, true, positive)
        correct = 2*3/7*3/4 / (3/7 + 3/4)
        self.assertTrue(score == correct)




