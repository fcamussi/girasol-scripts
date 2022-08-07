#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 30 01:49:50 2021

@author: fernando
"""

import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
from scipy.signal import find_peaks, peak_widths



def __centroid(cnt):
    M = cv.moments(cnt)
    cx = int(M['m10']/M['m00'])
    cy = int(M['m01']/M['m00'])
    return (cx,cy)


def __inBBx(cnt1, cnt2):
    [x1,_,w1,_] = cv.boundingRect(cnt1)
    [x2,_,w2,_] = cv.boundingRect(cnt2)
    return x2 <= x1 and x1+w1 <= x2+w2


COLORS = [(128,0,0),
          (0,128,0),
          (128,128,0),
          (0,0,128),
          (128,0,128),
          (0,128,128),
          (255,0,0),
          (0,255,0),
          (255,255,0),
          (0,0,255),
          (255,0,255),
          (0,255,255)]


fichero = 'bin4.png'
mask = cv.imread(fichero, cv.IMREAD_GRAYSCALE)
ppm = 143

# Perfiles acumulados (horizontalmente)
profiles = np.array(list(map(sum, mask)))

# Normalizo profiles entre 0 y 1
profiles = profiles / profiles.max()

# Busco máximos locales (centro de hileras) y sus anchos
row_centers,_ = find_peaks(profiles, prominence=0.1, distance=ppm/10)
widths = peak_widths(profiles, row_centers, rel_height=1/2)

plt.figure()
plt.plot(range(len(profiles)), profiles)
plt.xlabel('y')
plt.ylabel('perfil acumulado')
plt.savefig('bin4-perfiles.png', dpi=300)

plt.figure()
plt.plot(range(len(profiles)), profiles)
y = profiles[row_centers]
plt.plot(row_centers, y, 'x')
plt.hlines(*widths[1:], colors='g')
plt.xlabel('y')
plt.ylabel('perfil acumulado')
plt.savefig('bin4-maxs-widths.png', dpi=300)


widths = widths[0]

# Contornos de los objetos de la máscara
contours,_ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)

# Asigno cada objeto a un centro de hilera en base al umbral widths y el
# centroide del objeto en el eje y
rows = [[] for _ in range(len(row_centers))]
for i in range(len(rows)):
    for cnt in contours:
        cen = __centroid(cnt)  
        if (row_centers[i] - widths[i] < cen[1] and
            cen[1] < row_centers[i] + widths[i]):
            rows[i].append(cnt)

mask_rows = mask.copy()
mask_rows = cv.cvtColor(mask_rows, cv.COLOR_GRAY2RGB)
for i in range(len(rows)):
    cv.drawContours(mask_rows, rows[i], -1, COLORS[i%len(COLORS)], cv.FILLED)
cv.imwrite('bin4-paso1.png', mask_rows)

# Se eliminan los objetos que tienen las coordenadas x de su bounding
# box contenido en las coordenadas x del bounding box de otro objeto
rows2 = [[] for _ in range(len(rows))]
for i in range(len(rows)):
    for j in range(len(rows[i])):
        # Si el objeto a agregar no está completamente contenido en algún
        # objeto de row2...
        if len([x for x in rows2[i] if __inBBx(rows[i][j],x)]) == 0:
            e = [k for k in range(len(rows2[i]))
                     if __inBBx(rows2[i][k],rows[i][j])]
            # Si en row2 hay objetos completamente contenidos en el
            # objeto a agregar se eliminan
            if len(e) > 0:
                rows2[i] = [rows2[i][k] for k in range(len(rows2[i]))
                                if k not in e]
            # Se agrega el objeto
            rows2[i].append(rows[i][j])

mask_rows = mask.copy()
mask_rows = cv.cvtColor(mask_rows, cv.COLOR_GRAY2RGB)
for i in range(len(rows2)):
    cv.drawContours(mask_rows, rows2[i], -1, COLORS[i%len(COLORS)], cv.FILLED)
cv.imwrite('bin4-paso2.png', mask_rows)

# Para cada hilera se ordenan los objetos por posición x de su centroide
# de menor a mayor. Luego se filtran los objetos que están fuera del
# umbral widths/2 hacia arriba del centro de hilera y hacia abajo del
# mismo. Luego el umbral se va actualizando promediando los últimos 3
# objetos
rows3 = [[] for _ in range(len(rows2))]
for i in range(len(rows2)):
    rows2[i] = sorted(rows2[i], key=lambda x: __centroid(x)[0])
    for cnt in rows2[i]:
        cen = __centroid(cnt)
        if len(rows3[i]) < 3:
            rows3[i].append(cnt)
        else:
            row_centers[i] = sum([__centroid(c)[1] for c in
                                      rows3[i][-3:len(rows3[i])]])/3
            if (row_centers[i] - widths[i]/2 < cen[1] and
                cen[1] < row_centers[i] + widths[i]/2):
                rows3[i].append(cnt)

mask_rows = mask.copy()
mask_rows = cv.cvtColor(mask_rows, cv.COLOR_GRAY2RGB)
for i in range(len(rows3)):
    cv.drawContours(mask_rows, rows3[i], -1, COLORS[i%len(COLORS)], cv.FILLED)
cv.imwrite('bin4-paso3.png', mask_rows)

# Calcula las lineas de centroide a centroide
centroids = [list(map(__centroid, r)) for r in rows3]
centroids = [sorted(r, key=lambda x: x[0]) for r in centroids]
lines = [[list(cs[c]+cs[c+1]) for c in range(len(cs)-1)]
         for cs in centroids]
lines = [l for ls in lines for l in ls]

mask_rows = mask.copy()
mask_rows = cv.cvtColor(mask_rows, cv.COLOR_GRAY2RGB)
for l in lines:
    cv.line(mask_rows, (l[0],l[1]), (l[2],l[3]), (0,255,0), 2)
cv.imwrite('bin4-lineas.png', mask_rows)

contours2 = [x for r in rows3 for x in r]
total_rows = len(row_centers)
#print(len(contours))
#print(len(contours2))

