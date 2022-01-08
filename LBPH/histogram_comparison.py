from LBPH import *


class Matcher:
    def __init__(self):
        self.n_bins = 256

    def match(self, refs, lbp):
        best_score = float('inf')
        best_name = None
        hist = cv2.calcHist([lbp], [0], None, [256], [0, 256])
        hist /= hist.sum()

        for name, ref_hist in refs:
            score = cv2.compareHist(hist, ref_hist, cv2.HISTCMP_CHISQR)
            if score < best_score:
                best_score = score
                best_name = name
        best_score = best_score * 100
        return best_name, best_score
