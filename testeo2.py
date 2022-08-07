#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 21 07:34:54 2022

@author: fernando
"""

from descriptors import Descriptors
from counting_file import CountingFile
from rows_detection import rows_detection
import cv2 as cv
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score
import pickle
from PyQt5.QtCore import QRect


conteo_testeo = '../programa/camet2crx.counting'
conteo_inta = '../programa/conteo_INTA.csv'
coordenadas = '../programa/camet2crx-parcelas.csv'

parcelas = list(range(1, 41))
parcelas.remove(28)

counting_file = CountingFile(conteo_testeo)
df_coordenadas = pd.read_csv(coordenadas)

cv_mask = counting_file.getMask()
contours,_ = cv.findContours(cv_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
contours,total_rows,lines = rows_detection(cv_mask, counting_file.getPPM())

def cntInRect(cnt, rect):
    return rect.contains(QRect(*cv.boundingRect(cnt)))

dfs = []
muestra_parcela = []
for p in parcelas:
    rects = df_coordenadas.loc[df_coordenadas['parcela'] == p].values[:,1:]
    rects = list(map(lambda r: QRect(*r), rects))
    contours2 = [cnt for cnt in contours for r in rects if cntInRect(cnt, r)]
    df = Descriptors.compute(contours2, counting_file.getPPM())
    dfs.append(df)
    muestra_parcela.extend([p]*df.shape[0])
df = pd.concat(dfs)
print(f'Cantidad muestras testeo: {df.shape[0]}')
print('--- . ---')

data = df.values
X = data

# Estandarización
stsc_X = pickle.load(open('data/stsc_X', 'rb'))
stsc_y = pickle.load(open('data/stsc_y', 'rb'))
X = stsc_X.transform(X)

model_ls = pickle.load(open('data/model_ls', 'rb'))
model_ridge = pickle.load(open('data/model_ridge', 'rb'))
model_lasso = pickle.load(open('data/model_lasso', 'rb'))
model_svr = pickle.load(open('data/model_svr', 'rb'))

pred_ls = model_ls.predict(X)
pred_ridge = model_ridge.predict(X)
pred_lasso = model_lasso.predict(X)
pred_svr = model_svr.predict(X)

pred_ls = stsc_y.inverse_transform(pred_ls.reshape(-1,1))[:,0]
pred_ridge = stsc_y.inverse_transform(pred_ridge.reshape(-1,1))[:,0]
pred_lasso = stsc_y.inverse_transform(pred_lasso.reshape(-1,1))[:,0]
pred_svr = stsc_y.inverse_transform(pred_svr.reshape(-1,1))[:,0]

df_conteo_inta = pd.read_csv(conteo_inta)
testeo_ls = []
testeo_ridge = []
testeo_lasso = []
testeo_svr = []
testeo_inta = []
latex = """\\begin{figure}[H]
\\centering
\\begin{tabular}{cc}
\\includegraphics[width=120mm,valign=c]{scripts/camet2crx-parcela%d} &
\\begin{tabular}{l}Mín. cuad.: %.2f\\\\Ridge: %.2f\\\\Lasso: %.2f\\\\SVR: %.2f\\\\
INTA: %d \\end{tabular}
\\end{tabular}
\\caption{Conteo en parcela %d de camet2crx.tif}
\\end{figure}"""
for p in parcelas:
    muestras = np.where(np.array(muestra_parcela) == p)
    total_ls = sum(pred_ls[muestras])
    testeo_ls.append(total_ls)
    total_ridge = sum(pred_ridge[muestras])
    testeo_ridge.append(total_ridge)
    total_lasso = sum(pred_lasso[muestras])
    testeo_lasso.append(total_lasso)
    total_svr = sum(pred_svr[muestras])
    testeo_svr.append(total_svr)
    inta = df_conteo_inta.loc[df_conteo_inta['parcela'] == p].values[0][1]
    testeo_inta.append(inta)
    # print(f'#Parcela {p}#')
    # print(f'LS: {total_ls:.2f}')
    # print(f'Ridge={total_ridge:.2f}')
    # print(f'Lasso={total_lasso:.2f}')
    # print(f'SVR={total_svr:.2f}')
    # print(f'INTA: {inta}')
    print(latex % (p,total_ls,total_ridge,total_lasso,total_svr,inta,p))
print('--- . ---')

# Error a nivel parcelas
print('INTA:')
r2 = r2_score(testeo_inta, testeo_ls)
print('LS:')
print(f'R2: {r2:.2f}')
r2 = r2_score(testeo_inta, testeo_ridge)
print('Ridge:')
print(f'R2: {r2:.2f}')
r2 = r2_score(testeo_inta, testeo_lasso)
print('Lasso:')
print(f'R2: {r2:.2f}')
r2 = r2_score(testeo_inta, testeo_svr)
print('SVR:')
print(f'R2: {r2:.2f}')
