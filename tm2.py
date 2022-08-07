#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 12 19:19:42 2021

@author: fernando
"""

import cv2 as cv
from scipy.signal import correlate

G_masked = cv.imread('img4-G-masked.png', cv.IMREAD_GRAYSCALE)
template = cv.imread('template2.png', cv.IMREAD_GRAYSCALE)

G_masked = cv.normalize(G_masked, None, 0, 1, cv.NORM_MINMAX, cv.CV_32F)
template = cv.normalize(template, None, 0, 1, cv.NORM_MINMAX, cv.CV_32F)
xc = correlate(G_masked, template, mode = 'full', method='fft')
xc = cv.normalize(xc, None, 0, 255, cv.NORM_MINMAX, cv.CV_8U)

cv.imwrite('img4-G-masked-corr2.png', xc)
