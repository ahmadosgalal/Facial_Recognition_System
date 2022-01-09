import math as m
import cv2
import numpy as np
from skimage.feature import local_binary_pattern
from matplotlib import pyplot as plt


class LBPHfromScratch:
    def __init__(self):
        self.R = 1
        self.P = 8
        self.filter_size = 3
        # Anti-clockwise (right -> up + right -> up -> up + left -> left -> down + left -> down -> down + right)
        self.filter_lbp = np.array([[2, 1], [2, 0], [1, 0], [0, 0], [0, 1], [0, 2], [1, 2], [2, 2]])

    def Compute_LBP(self, img):
        # Determine the dimensions of the input image.
        height = img.shape[0]
        width = img.shape[1]

        if width < self.filter_size or height < self.filter_size:
            print("Size not correct!")
            return

        out_width = width - self.filter_size + 1
        out_height = height - self.filter_size + 1

        reference_matrix = img[1:1 + out_height, 1:1 + out_width]

        out_img = np.zeros((out_height, out_width))

        for i in range(0, 8):
            step_x, step_y = self.filter_lbp[i]

            sliding_matrix = img[step_y:step_y + out_height, step_x:step_x + out_width]

            flags = np.greater_equal(sliding_matrix, reference_matrix)

            exponent = np.power(2, i)
            out_img = out_img + (flags * exponent)

        return out_img.astype(np.uint8)
