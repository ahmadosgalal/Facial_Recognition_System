import math as m
import cv2
import numpy as np
from skimage.feature import local_binary_pattern
from matplotlib import pyplot as plt


class LBPReady:
    def __init__(self, nb_points, radius):
        # Initiate number of points(neighbors) and the radius of the cercle
        self._Nb_Points = nb_points
        self._Radius = radius

    @property
    def radius(self):
        return self._Radius

    @property
    def nb_points(self):
        return self._Nb_Points

    def compute(self, img):
        # compute the Local Binary Pattern of the image,
        # and then use the LBP representation
        # to build the histogram of patterns
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img1 = np.array(rgb_img, 'uint8')
        gray = np.array(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 'uint8')

        fig, axs = plt.subplots(2, 2, figsize=(80, 80), facecolor='w', edgecolor='k')
        fig.subplots_adjust(hspace=.5, wspace=.001)

        gray = lambda rgb: np.dot(rgb[..., :3], [0.299, 0.587, 0.114])
        gray = gray(img1)
        axs[0][0].imshow(img1, cmap='gray', vmin=0, vmax=255)
        axs[0][0].set_title('Original Image', fontdict={'fontsize': 15, 'fontweight': 'medium'})
        axs[0][0].axis('off')

        axs[0][1].imshow(gray, cmap='gray', vmin=0, vmax=255)
        axs[0][1].set_title('GrayScale Image', fontdict={'fontsize': 15, 'fontweight': 'medium'})
        axs[0][1].axis('off')

        LBP = local_binary_pattern(gray, self._Nb_Points, self._Radius, method="uniform")
        axs[1][1].imshow(LBP, cmap='gray', vmin=0, vmax=9)
        axs[1][1].set_title('LBP Image', fontdict={'fontsize': 15, 'fontweight': 'medium'})
        axs[1][1].axis('off')
        (hist, bins) = np.histogram(LBP.ravel(),
                                    bins=np.arange(0, self._Nb_Points + 3),
                                    range=(0, self._Nb_Points + 2))
        width = bins[1] - bins[0]
        center = (bins[:-1] + bins[1:]) / 2

        axs[1][0].bar(center, hist, align='center', width=width)
        axs[1][0].set_title('Histogram', fontdict={'fontsize': 15, 'fontweight': 'medium'})
        # normalize the histogram
        hist = hist.astype("float")
        hist /= hist.sum()

        plt.show()
        return hist
