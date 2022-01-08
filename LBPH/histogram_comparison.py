"""
===============================================
Local Binary Pattern for texture classification
===============================================

In this example, we will see how to classify textures based on LBP (Local
Binary Pattern). The histogram of the LBP result is a good measure to classify
textures. For simplicity the histogram distributions are then tested against
each other using the Kullback-Leibler-Divergence.
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import scipy.ndimage as nd
import skimage.feature as ft
from skimage import data
from LBPH import *


class Matcher:
    def __init__(self, p=8, r=1):
        # settings for LBP
        self.METHOD = 'uniform'
        self.P = p
        self.R = r
        self.n_bins = 256

    def kullback_leibler_divergence(self, p, q):
        p = np.asarray(p)
        q = np.asarray(q)
        filt = np.logical_and(p != 0, q != 0)
        return np.sum(p[filt] * np.log2(p[filt] / q[filt]))

    def match(self, refs, lbp):
        best_score = float('inf')
        best_name = None

        hist, _ = np.histogram(lbp, density=True, bins=self.n_bins)
        for name, ref in refs:
            ref_hist, _ = np.histogram(ref, density=True, bins=self.n_bins)
            score = self.kullback_leibler_divergence(hist, ref_hist)
            if np.abs(score) < best_score:
                best_score = np.abs(score)
                best_name = name
            # print(name, score)
        return best_name, best_score
