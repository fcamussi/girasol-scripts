#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 18 07:34:14 2022

@author: fernando
"""

from descriptors import Descriptors
from counting_file import CountingFile
import cv2 as cv
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import pickle
from PyQt5.QtCore import QRect


conteo_testeo = '../programa/camet1x-testeo.counting'
conteo_inta = '../programa/conteo_INTA.csv'
coordenadas = '../programa/camet1x-parcelas-testeo.csv'

# parcelas de testeo
_,parcelas = train_test_split(range(1, 151), test_size=0.1, random_state=0)
print(f'parcelas testeo: {parcelas}')
print('--- . ---')

counting_file = CountingFile(conteo_testeo)
points = counting_file.getPoints()
df_coordenadas = pd.read_csv(coordenadas)

cv_mask = counting_file.getMask()
contours,_ = cv.findContours(cv_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)

dfs = []
muestra_parcela = []
for p in parcelas:
    rects = df_coordenadas.loc[df_coordenadas['parcela'] == p].values[:,1:]
    rects = list(map(lambda r: QRect(*r), rects))
    points2 = [p for p in points for r in rects if r.contains(p)]
    df = Descriptors.compute(contours, counting_file.getPPM(), points2)
    dfs.append(df)
    muestra_parcela.extend([p]*df.shape[0])
df = pd.concat(dfs)
print(f'Cantidad muestras testeo: {df.shape[0]}')
print('--- . ---')

data = df.values
X = data[:,:-1]
y = data[:,-1]

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

r2 = r2_score(y, pred_ls)
print('LS:')
print(f'R2: {r2:.2f}')
r2 = r2_score(y, pred_ridge)
print('Ridge:')
print(f'R2: {r2:.2f}')
r2 = r2_score(y, pred_lasso)
print('Lasso:')
print(f'R2: {r2:.2f}')
r2 = r2_score(y, pred_svr)
print('SVR:')
print(f'R2: {r2:.2f}')
print('--- . ---')

df_conteo_inta = pd.read_csv(conteo_inta)
testeo_ls = []
testeo_ridge = []
testeo_lasso = []
testeo_svr = []
testeo_inta = []
parcelas = np.sort(parcelas)
latex = """\\begin{figure}[H]
\\centering
\\begin{tabular}{cc}
\\includegraphics[width=120mm,valign=c]{scripts/camet1x-parcela%d} &
\\begin{tabular}{l}Mín. cuad.: %.2f\\\\Ridge: %.2f\\\\Lasso: %.2f\\\\SVR: %.2f\\\\
Marcadas: %d\\\\INTA: %d \\end{tabular}
\\end{tabular}
\\caption{Conteo en parcela %d de camet1x.tif}
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
    marcadas = sum(y[muestras])
    inta = df_conteo_inta.loc[df_conteo_inta['parcela'] == p].values[0][1]
    testeo_inta.append(inta)
    # print(f'#Parcela {p}#')
    # print(f'LS: {total_ls:.2f}')
    # print(f'Ridge={total_ridge:.2f}')
    # print(f'Lasso={total_lasso:.2f}')
    # print(f'SVR={total_svr:.2f}')
    # print(f'marcadas: {marcadas:.0f}')
    # print(f'INTA: {inta}')
    print(latex % (p,total_ls,total_ridge,total_lasso,total_svr,marcadas,inta,p))
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
