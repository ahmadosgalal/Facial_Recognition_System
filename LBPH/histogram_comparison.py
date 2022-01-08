"""
===============================================
Local Binary Pattern for texture classification
===============================================

In this example, we will see how to classify textures based on LBP (Local
Binary Pattern). The histogram of the LBP result is a good measure to classify
textures. For simplicity the histogram distributions are then tested against
each other using the Kullback-Leibler-Divergence.
"""
from LBPH import *


class Matcher:
    def __init__(self):
        self.n_bins = 256

    def match(self, refs, lbp):
        best_score = float('inf')
        best_name = None
        hist = cv2.calcHist([lbp], [0], None, [256], [0, 256])
        hist /= hist.sum()

        for name, ref in refs:
            ref_hist = cv2.calcHist([ref], [0], None, [256], [0, 256])
            ref_hist /= ref_hist.sum()

            score = cv2.compareHist(hist, ref_hist, cv2.HISTCMP_CHISQR)
            if score < best_score:
                best_score = score
                best_name = name
        best_score = best_score * 100
        return best_name, best_score
