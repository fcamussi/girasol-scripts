#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 27 04:16:28 2021

@author: fernando
"""

import cv2 as cv
import numpy as np


fichero = 'bin2.png'

mask = cv.imread(fichero, cv.IMREAD_GRAYSCALE)

ppm = 143
U = (ppm/20)**2
contours,_ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
contours = list(filter(lambda a: cv.contourArea(a) > U, contours))
mask2 = np.zeros(mask.shape, np.uint8)
for cnt in contours: cv.drawContours(mask2, [cnt], 0, 255, cv.FILLED)

cv.imwrite('bin2-fill-remove.png', mask2)
