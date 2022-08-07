#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  3 07:45:32 2021

@author: fernando
"""

from counting_file import CountingFile
from rows_detection import rows_detection
import cv2 as cv


file = '../programa/camet1x-completo.counting'

counting_file = CountingFile(file)
mask = counting_file.getMask()
points = counting_file.getPoints()
contours,_ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)

isIn = lambda p,cnt: cv.pointPolygonTest(cnt, (p.x(),p.y()), False) >= 0

def calcular():
    # Grupos de plantas detectados
    TP = [cnt for cnt in contours if [p for p in points if isIn(p, cnt)]]
    # Grupos de plantas no detectados
    FN = [p for p in points if not [cnt for cnt in contours if isIn(p, cnt)]]
    # Grupos de plantas detectados que no son plantas
    FP = [cnt for cnt in contours if not [p for p in points if isIn(p, cnt)]]
    TP = len(TP)
    FN = len(FN)
    FP = len(FP)
    print(f'TP: {TP}, FN: {FN}, FP: {FP}')
    precision = TP/(TP+FP)
    recall = TP/(TP+FN)
    print(f'Precision: {precision:.2f}, recall: {recall:.2f}')

print('Sin detección de hileras:')
calcular()
contours,_,_ = rows_detection(mask, counting_file.getPPM())
print('Con detección de hileras:')
calcular()
