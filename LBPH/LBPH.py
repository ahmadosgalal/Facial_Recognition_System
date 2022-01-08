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


class LBPHfromScratch:
    def __init__(self):
        self.R = 1
        self.P = 8
        self.filter_size = 3
        # Anti-clockwise (right -> up + right -> up -> up + left -> left -> down + left -> down -> down + right)
        self.filter_lbp = np.array([[2, 1], [2, 0], [1, 0], [0, 0], [0, 1], [0, 2], [1, 2], [2, 2]])
        # self.filter_lbp = np.array([[2, 1], [2, 2], [1, 2], [0, 2], [0, 1], [0, 0], [1, 0], [2, 0]])

    def Compute_LBP(self, img):
        # Determine the dimensions of the input image.
        height = img.shape[0]
        width = img.shape[1]

        print(height, width, "##")

        # Minimum allowed size for the input image depends on the radius of the used LBP operator.
        if width < self.filter_size or height < self.filter_size:
            raise Exception('Too small input image. Should be at least (2*radius+1) x (2*radius+1)')

        out_width = width - self.filter_size + 1
        out_height = height - self.filter_size + 1

        print("##", out_width, out_height)

        # Fill the center pixel matrix C.
        C = img[1:1 + out_height, 1:1 + out_width]

        print("C.shape", C.shape)

        # Initialize the result matrix with zeros.
        out_img = np.zeros((out_height, out_width), dtype=np.float32)

        for i in range(0, 8):

            print("i", i)

            rx = self.filter_lbp[i][0]
            ry = self.filter_lbp[i][1]

            print("rx, ry:", rx, ry)

            N = img[ry:ry + out_height, rx:rx + out_width]

            print("N.shape:", N.shape)

            D = (N >= C).astype(np.uint8)

            # Update the result matrix.
            v = 2 ** i
            out_img += D * v

        return out_img.astype(np.uint8)


class LBPbyHand:
    def __init__(self, neighbors=8, radius=1):
        self._radius = radius
        self._neighbors = neighbors

    @property
    def radius(self):
        return self._radius

    @property
    def neighbors(self):
        return self._neighbors

    def Compute_LBP(self, Image):
        # Determine the dimensions of the input image.
        ysize, xsize = Image.shape
        # define circle of symetrical neighbor points
        angles_array = 2 * np.pi / self._neighbors
        alpha = np.arange(0, 2 * np.pi, angles_array)
        # Determine the sample points on circle with radius R
        s_points = np.array([-np.sin(alpha), np.cos(alpha)]).transpose()
        s_points *= self._radius
        # s_points is a 2d array with 2 columns (y,x) coordinates for each cicle neighbor point		
        # Determine the boundaries of s_points wich gives us 2 points of coordinates
        # gp1(min_x,min_y) and gp2(max_x,max_y), the coordinate of the outer block 
        # that contains the circle points
        min_y = min(s_points[:, 0])
        max_y = max(s_points[:, 0])
        min_x = min(s_points[:, 1])
        max_x = max(s_points[:, 1])
        # Block size, each LBP code is computed within a block of size bsizey*bsizex
        # so if radius = 1 then block size equal to 3*3
        bsizey = np.ceil(max(max_y, 0)) - np.floor(min(min_y, 0)) + 1
        bsizex = np.ceil(max(max_x, 0)) - np.floor(min(min_x, 0)) + 1
        # Coordinates of origin (0,0) in the block
        origy = int(0 - np.floor(min(min_y, 0)))
        origx = int(0 - np.floor(min(min_x, 0)))
        # Minimum allowed size for the input image depends on the radius of the used LBP operator.
        if xsize < bsizex or ysize < bsizey:
            raise Exception('Too small input image. Should be at least (2*radius+1) x (2*radius+1)')
        # Calculate dx and dy: output image size
        # for exemple, if block size is 3*3 then we need to substract the first row and the last row which is 2 rows
        # so we need to substract 2, same analogy applied to columns
        dx = int(xsize - bsizex + 1)
        dy = int(ysize - bsizey + 1)
        # Fill the center pixel matrix C.
        C = Image[origy:origy + dy, origx:origx + dx]
        # Initialize the result matrix with zeros.
        result = np.zeros((dy, dx), dtype=np.float32)
        for i in range(s_points.shape[0]):
            # Get coordinate in the block:
            p = s_points[i][:]
            y, x = p + (origy, origx)
            # Calculate floors, ceils and rounds for the x and ysize
            fx = int(np.floor(x))
            fy = int(np.floor(y))
            cx = int(np.ceil(x))
            cy = int(np.ceil(y))
            rx = int(np.round(x))
            ry = int(np.round(y))
            D = [[]]
            if np.abs(x - rx) < 1e-6 and np.abs(y - ry) < 1e-6:
                # Interpolation is not needed, use original datatypes
                N = Image[ry:ry + dy, rx:rx + dx]
                D = (N >= C).astype(np.uint8)
            else:
                # interpolation is needed
                # compute the fractional part.
                ty = y - fy
                tx = x - fx
                # compute the interpolation weight.
                w1 = (1 - tx) * (1 - ty)
                w2 = tx * (1 - ty)
                w3 = (1 - tx) * ty
                w4 = tx * ty
                # compute interpolated image:
                N = w1 * Image[fy:fy + dy, fx:fx + dx]
                N = np.add(N, w2 * Image[fy:fy + dy, cx:cx + dx], casting="unsafe")
                N = np.add(N, w3 * Image[cy:cy + dy, fx:fx + dx], casting="unsafe")
                N = np.add(N, w4 * Image[cy:cy + dy, cx:cx + dx], casting="unsafe")
                D = (N >= C).astype(np.uint8)
            # Update the result matrix.
            v = 2 ** i
            result += D * v
        return result.astype(np.uint8)
