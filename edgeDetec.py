'''
	file: 		edgeDetect.py
	authors: 	[jvitti, italijancic, mbassani, iprieto]
	date:		05/05/2022
	content:	Try to do preliminary procces of image using opencv2 library
'''

# Python package imports
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from fitting import elipseFit


#
# Load image file from system path
# --------------------------------
img = cv2.imread('./img/test.jpeg', 0)
# img = cv2.imread('./img/test.jpeg', cv2.IMREAD_ANYDEPTH)

# Get edges of image
edges = cv2.Canny(img, 127, 255)     #--- image containing edges ---

# Show original img and filtered
# plt.figure()
# plt.plot()
# plt.title('Original')
# plt.imshow(img)

# plt.figure()
# plt.plot()
# plt.title('cv2.Canny')
# plt.imshow(edges)

#
# Get coordinates of edge line
# ----------------------------
indexs = np.asarray(np.where(edges != [0]))		# np.where(edges != 0) --> Return a tuple

plt.figure()
# plt.plot(indexs[1,:], (indexs[0,:]), '.w')
plt.imshow(edges)
plt.title('Edge from np.array')
plt.plot(indexs[1,0:350], (indexs[0,0:350]), '.w')
# plt.plot(indexs[1,:], indexs[0,:], '.w')
# xe, ye, coefs = elipseFit(indexs[1,0:350], indexs[0,0:350])
xe, ye, coefs = elipseFit(indexs[1,0:100], indexs[0,0:100])
# xe, ye, coefs = elipseFit(indexs[1,:], indexs[0,:])
print('x0, y0, ap, bp, e, phi = ', coefs)
plt.plot(xe, ye, '.r')

if 'PYCONTROL_TEST_EXAMPLES' not in os.environ:
    plt.show()