#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 22 15:31:47 2021

@author: fernando
"""

import pandas as pd
import matplotlib.pyplot as plt


file = './data/camet1-entrenamiento-testeo-desc.csv'
df = pd.read_csv(file)
X = df[df.columns[:-1]].values
y = df[df.columns[-1]].values

# Histograma de CANTIDAD_PLANTAS
plt.hist(y, bins=int(max(y)))
plt.xlim(min(y), max(y))
plt.xlabel('$CANTIDAD\_PLANTAS$')
plt.savefig('hist-plantas.png', dpi=300)

print(f'Muestras totales: {df.shape[0]}')
