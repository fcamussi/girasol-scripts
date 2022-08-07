#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  3 02:59:21 2021

@author: fernando
"""

from skimage.feature import hog
from skimage import exposure
import cv2 as cv


image = cv.imread('img3.png', cv.IMREAD_COLOR)
mask = cv.imread('img3-bin.png', cv.IMREAD_GRAYSCALE)

gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
masked = gray * (mask>0)

fd, hog = hog(masked, orientations=9, pixels_per_cell=(8, 8),
              cells_per_block=(2, 2), visualize=True)

hog = exposure.rescale_intensity(hog, in_range=(0, 10))
hog = cv.normalize(hog, None, 0, 255, cv.NORM_MINMAX, cv.CV_8U)

hog = cv.addWeighted(masked, 1, hog, 1, 0)

cv.imwrite('img3-masked.png', masked)
cv.imwrite('img3-hog.png', hog)
