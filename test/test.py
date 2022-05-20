"""Tests for `sktools` package."""

import unittest

import pandas as pd
import numpy as np
from ATC_opt import OptimizedRounder
from sklearn.metrics import accuracy_score

class Rounder(unittest.TestCase):
    """Tests"""

    def setUp(self):
        """Create dataframe with categories and a target variable"""

        self.a = np.array([0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1])
        self.b = np.array([0,0,0,0,1,1,1,1,1,1])
        self.c = np.array([0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1])-0.1
        self.d = np.array([0,1,1,1,1,1,1,1,1,1])



    def test_one(self):
        """
        Expected output of percentile 50 in df:
            - a median is 4 (a values are 1, 4, 6)
            - b median is 5 (b values are 2, 5, 7)
            - c median is 0 (c values are 0)
        """

        opt = OptimizedRounder()
        opt.fit(self.c,self.b)


        np.testing.assert_array_equal(
            accuracy_score(self.b,opt.predict(self.c,opt.coefficients())),1
        )
