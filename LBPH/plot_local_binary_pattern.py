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


class Matcher:
    def __init__(self, p=8, r=1):
        # settings for LBP
        self.METHOD = 'uniform'
        self.P = p
        self.R = r
        #self.matplotlib.rcParams['font.size'] = 9

    def kullback_leibler_divergence(self, p, q):
        p = np.asarray(p)
        q = np.asarray(q)
        filt = np.logical_and(p != 0, q != 0)
        return np.sum(p[filt] * np.log2(p[filt] / q[filt]))

    def match(self, refs, img):
        best_score = 10
        best_name = None
        lbp = ft.local_binary_pattern(img, self.P, self.R, self.METHOD)
        hist, _ = np.histogram(lbp,  bins=self.P + 2, range=(0, self.P + 2))
        for name, ref in refs:
            ref_hist, _ = np.histogram(ref,  bins=self.P + 2,
                                       range=(0, self.P + 2))
            score = self.kullback_leibler_divergence(hist, ref_hist)
            if np.abs(score) < best_score:
                best_score = np.abs(score)
                best_name = name
            print(name, score)
        return best_name, best_score
