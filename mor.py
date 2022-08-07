#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 27 04:16:28 2021

@author: fernando
"""

import cv2 as cv


fichero = 'bin1.png'

mask = cv.imread(fichero, cv.IMREAD_GRAYSCALE)

kernel = cv.getStructuringElement(shape=cv.MORPH_ELLIPSE, ksize=(5,5))
mask_open = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel)

cv.imwrite('bin1-open.png', mask_open)
