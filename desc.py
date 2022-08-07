#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 10 09:08:14 2021

@author: fernando
"""

import cv2 as cv
import numpy as np


fichero = 'bin7.png'

mask = cv.imread(fichero, cv.IMREAD_GRAYSCALE)
mask = cv.resize(mask, (mask.shape[1]*10, mask.shape[0]*10),
                 interpolation = cv.INTER_AREA)

contour,_ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
contour = contour[0]

cnt_x = contour[:,0][:,0]
cnt_y = contour[:,0][:,1]
hull = cv.convexHull(np.vstack((cnt_x,cnt_y)).T)

image = mask.copy()
image = cv.cvtColor(image, cv.COLOR_GRAY2RGB)

cv.polylines(image, [hull], True, (0,0,255), 8)

cv.imwrite('bin7-resized.png', mask)
cv.imwrite('bin7-hull.png', image)
