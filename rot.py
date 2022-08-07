#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 12 19:19:42 2021

@author: fernando
"""

import cv2 as cv
from rotation import rotation

fichero = 'seg2-rgb.png'
image = cv.imread(fichero, cv.IMREAD_COLOR)

orientation = 42.09534878060612
rotated_image = rotation(image, -orientation)
cv.imwrite('seg2-rgb-rot.png', rotated_image)
