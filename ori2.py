#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 12 19:19:42 2021

@author: fernando
"""

import cv2 as cv
import numpy as np

fichero = 'img2.png'
image = cv.imread(fichero, cv.IMREAD_COLOR)

e1 = np.array([0.74203026, 0.67036638])*300
orientation = 42.09534878060612

[h,w,_] = image.shape

x1 = int(w/2)-100
y1 = int(h/2)+100

p1 = (x1,y1)
p2 = (int(x1+e1[0]),int(y1-e1[1]))

cv.arrowedLine(image, p1, p2, (0,0,255), 10, cv.LINE_AA, 0, 0.3)
cv.imwrite('img2-arrow.png', image)
