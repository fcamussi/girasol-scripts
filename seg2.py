#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 27 04:16:28 2021

@author: fernando
"""

import cv2 as cv
import numpy as np


fichero = 'img2.png'

image = cv.imread(fichero, cv.IMREAD_COLOR)

R = np.copy(image[:,:,2])
G = np.copy(image[:,:,1])
B = np.copy(image[:,:,0])

R=R.astype(int)
G=G.astype(int)
B=B.astype(int)
ExG = 2*G-R-B
ExG[ExG>255]=255
ExG[ExG<0]=0
ExG=ExG.astype('uint8')

_,ExG_bin = cv.threshold(ExG, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
cv.imwrite('img2-bin.png', ExG_bin)
