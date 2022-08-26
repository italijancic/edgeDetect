import numpy as np


def twoDotsLineFit(x, y):
	xi = x[0]
	xf = x[1]
	yi = y[0]
	yf = y[1]
	# xi, xf = x
	# yi, yf = y

    # Fit line to user selected points
	m = (yf - yi) / (xf - xi)
	b = yi - m * xi

	return [m, b]

