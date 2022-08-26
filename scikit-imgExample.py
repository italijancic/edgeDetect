'''
	file: 		edgeDetect.py
	authors: 	[jvitti, italijancic, mbassani, iprieto]
	date:		05/05/2022
	content:	Try to do preliminary proccess of image using sckit-imgae library
'''

import os
from skimage import data, io, filters
from skimage.filters import threshold_otsu
from skimage.color import rgb2gray
import matplotlib.pyplot as plt
import numpy as np

# Load img from file system
image_rgb = io.imread('./img/test.jpeg')

# pass to gray scale
image = rgb2gray(image_rgb)

# gray scale to binary image
thresh = threshold_otsu(image)
binary = image > thresh

# Get edges
edges = filters.sobel(binary)

#
# Show image
# ----------
# plt.figure()
# plt.title('edges filters.sobel')
# plt.imshow(edges)

# plt.figure()
# plt.imshow(image, cmap=plt.cm.gray)
# plt.title('Original Gray Scale')

# plt.figure()
# plt.imshow(binary, cmap=plt.cm.gray)
# plt.title('Original Binarized')

# Get edges coordinates as np.array oject
indexs = np.asarray(np.where(edges != [0]))

# Plot edges coordinates
# plt.figure()
# plt.plot(indexs[1,:], (indexs[0,:]), '.b')
# plt.imshow(image, cmap=plt.cm.gray)
# plt.title('Detected Edge Coordinates')

plt.figure()
plt.plot(indexs[1,:], (indexs[0,:]), '.r')
plt.imshow(binary, cmap=plt.cm.gray)

if 'PYCONTROL_TEST_EXAMPLES' not in os.environ:
    plt.show()