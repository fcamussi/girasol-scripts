#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 12 19:19:42 2021

@author: fernando
"""

import cv2 as cv
import numpy as np

image = cv.imread('img4.png', cv.IMREAD_COLOR)
mask = cv.imread('img4-bin.png', cv.IMREAD_GRAYSCALE)

R = np.copy(image[:,:,2])
G = np.copy(image[:,:,1])
B = np.copy(image[:,:,0])

mask = cv.normalize(mask, None, 0, 1, cv.NORM_MINMAX, cv.CV_32F)
G_masked = G * mask;
G_masked = cv.normalize(G_masked, None, 0, 255, cv.NORM_MINMAX, cv.CV_8U)

cv.imwrite('img4-G-masked.png', G_masked)
