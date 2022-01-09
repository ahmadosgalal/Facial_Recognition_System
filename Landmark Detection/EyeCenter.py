import numpy as np
from scipy.ndimage.filters import gaussian_filter

class GradientIntersect:

	def createGrid(self, Y, X):
		grid = np.array([[y,x] for y in range(1-Y,Y) for x in range (1-X,X)], dtype='float')
		grid2 = np.sqrt(np.einsum('ij,ij->i',grid,grid))
		grid2[grid2==0] = 1
		grid /= grid2[:,np.newaxis]
		grid = grid.reshape(Y*2-1,X*2-1,2)
		return grid

	def createGradient(self, image):
		Y, X = image.shape
		gy, gx = np.gradient(image)
		gx = np.reshape(gx, (X*Y,1))
		gy = np.reshape(gy, (X*Y,1))
		gradient = np.hstack((gy,gx))
		gradient2 = np.sqrt(np.einsum('ij,ij->i',gradient,gradient))
		gradient2[gradient2==0] = 1
		gradient /= gradient2[:,np.newaxis]
		return gradient

	def locate(self, image, sigma = 2, accuracy = 1):
		Y, X = image.shape
		grid = self.createGrid(Y, X)
		scores = np.zeros((Y,X))
		image = (image.astype('float') - np.min(image)) / np.max(image)
		blurred = gaussian_filter(image, sigma=sigma)
		gradient = self.createGradient(image)
		for cy in range(0,Y,accuracy):
			for cx in range(0,X,accuracy):
				displacement = grid[Y-cy-1:Y*2-cy-1,X-cx-1:X*2-cx-1,:]
				displacement = np.reshape(displacement,(X*Y,2))
				score = np.einsum('ij,ij->i', displacement, gradient)
				score = np.einsum('i,i->', score, score)
				scores[cy,cx] = score
		scores = scores * (1-blurred)
		if accuracy>1:
			(yval,xval) = np.unravel_index(np.argmax(scores),scores.shape)
			yval = min(max(yval,accuracy), Y-accuracy-1)
			xval = min(max(xval,accuracy), X-accuracy-1)
			for cy in range(yval-accuracy,yval+accuracy+1):
				for cx in range(xval-accuracy,xval+accuracy+1):
					displacement = grid[Y-cy-1:Y*2-cy-1,X-cx-1:X*2-cx-1,:]
					displacement = np.reshape(displacement,(X*Y,2))
					score = np.einsum('ij,ij->i', displacement, gradient)
					score = np.einsum('i,i->', score, score)
					scores[cy,cx] = score * (1-blurred[cy,cx])
		(yval,xval) = np.unravel_index(np.argmax(scores),scores.shape)	
		return (yval, xval)

	def track(self, image, prev, sigma = 2, radius=50, distance = 10):
		py, px = prev
		image = image[py-radius:py+radius+1, px-radius:px+radius+1]
		Y, X = image.shape
		grid = self.createGrid(Y, X)
		scores = np.zeros((Y,X))
		image = (image.astype('float') - np.min(image)) / np.max(image)
		blurred = gaussian_filter(image, sigma=sigma)
		gradient = self.createGradient(image)
		for cy in range(radius-distance, radius+distance+1):
			for cx in range(radius-distance, radius+distance+1):
				displacement = grid[Y-cy-1:Y*2-cy-1,X-cx-1:X*2-cx-1,:]
				displacement = np.reshape(displacement,(X*Y,2))
				score = np.einsum('ij,ij->i', displacement, gradient)
				score = np.einsum('i,i->', score, score)
				scores[cy,cx] = score
		scores = scores * (1-blurred)
		(yval,xval) = np.unravel_index(np.argmax(scores),scores.shape)
		return (py+yval-radius, px+xval-radius)
	

class IsophoteCurvature:

	def __init__(self, blur = 3, minrad = 2, maxrad = 20):
		self.blur = blur
		self.minrad = minrad
		self.maxrad = maxrad

	def locate(self, image):
		image = gaussian_filter(image, sigma=self.blur)
		image = (image.astype('float') - np.min(image))
		image = image / np.max(image)
		Ly, Lx = np.gradient(image)
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
		center_map = np.zeros(image.shape, image.dtype)
		(height, width)=center_map.shape   
		for y in range(height):
			for x in range(width):
				if Dx[y][x] == 0 and Dy[y][x] == 0:
					continue
				if (x + Dx[y][x])>0 and (y + Dy[y][x])>0:
					if (x + Dx[y][x]) < center_map.shape[1] and (y + Dy[y][x]) < center_map.shape[0] and k[y][x]<0:
						if displacement[y][x] >= self.minrad and displacement[y][x] <= self.maxrad:
							center_map[int(y+Dy[y][x])][int(x+Dx[y][x])] += curvedness[y][x]
		center_map = gaussian_filter(center_map, sigma=self.blur)
		blurred = gaussian_filter(image, sigma=self.blur)
		center_map = center_map * (1-blurred)
		position = np.unravel_index(np.argmax(center_map), center_map.shape)
		return position