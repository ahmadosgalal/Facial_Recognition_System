import numpy as np
from scipy.ndimage.filters import gaussian_filter

class EyeCenterLocator:

	def __init__(self, blur = 3, minrad = 2, maxrad = 20):
		self.blur = blur
		self.minrad = minrad
		self.maxrad = maxrad

	def locate(self, image_BGR):
		image_BGR = gaussian_filter(image_BGR, sigma=self.blur)
		image_BGR = (image_BGR.astype('float') - np.min(image_BGR))
		image_BGR = image_BGR / np.max(image_BGR)
		
		Ly, Lx = np.gradient(image_BGR)
		Lyy, Lyx = np.gradient(Ly)
		Lxy, Lxx = np.gradient(Lx)
		Lvv = Ly**2 * Lxx - 2*Lx * Lxy * Ly + Lx**2 * Lyy
		Lw =  Lx**2 + Ly**2
		Lw[Lw==0] = 0.001
		Lvv[Lvv==0] = 0.001
		k = - Lvv / (Lw**1.5)
		
		Dx =  -Lx * (Lw / Lvv)
		Dy =  -Ly * (Lw / Lvv)
		displacement = np.sqrt(Dx**2 + Dy**2)
		
		curvedness = np.absolute(np.sqrt(Lxx**2 + 2 * Lxy**2 + Lyy**2))
		center_map = np.zeros(image_BGR.shape, image_BGR.dtype)
		for i in range(center_map.shape[0]):
			for j in range(center_map.shape[1]):
				if Dx[i][j] == 0 and Dy[i][j] == 0:
					continue
				if (j + Dx[i][j])>0 and (i + Dy[i][j])>0:
					if (j + Dx[i][j]) < center_map.shape[1] and (i + Dy[i][j]) < center_map.shape[0] and k[i][j]<0:
						if displacement[i][j] >= self.minrad and displacement[i][j] <= self.maxrad:
							center_map[int(i+Dy[i][j])][int(j+Dx[i][j])] += curvedness[i][j]
		center_map = gaussian_filter(center_map, sigma=self.blur)
		blurred = gaussian_filter(image_BGR, sigma=self.blur)
		center_map = center_map * (1-blurred)
		position = np.unravel_index(np.argmax(center_map), center_map.shape)
		return position