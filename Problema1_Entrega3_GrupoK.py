# -*- coding: utf-8 -*-
"""
Created on Sat Nov 30 18:37:53 2024

@author: Lucía Campos
        Lucía Varona
        Juan Francisco García
        David Rodríguez
"""

import pandas as pd
import pulp as lp

problema = lp.LpProblem("Asignaciondequirofanos", lp.LpMinimize)

operaciones_df = pd.read_excel('C:/Users/Usuario/Downloads/241204_datos_operaciones_programadas.xlsx')
costes_df = pd.read_excel('C:/Users/Usuario/Downloads/241204_costes.xlsx')

operaciones_cardio_df = operaciones_df[operaciones_df["Especialidad quirúrgica"] == "Cardiología Pediátrica"]

quirofanos = costes_df['Unnamed: 0'].tolist()
operaciones_cardio = operaciones_cardio_df['Código operación'].tolist()

incompatibilidades = {}
for i, op1 in operaciones_cardio_df.iterrows():
    incompatibilidades[op1['Código operación']] = []
    for i2, op2 in operaciones_cardio_df.iterrows():
        if op1['Código operación'] != op2['Código operación']:
            if not (op1['Hora fin'] <= op2['Hora inicio '] or op1['Hora inicio '] >= op2['Hora fin']):
                incompatibilidades[op1['Código operación']].append(op2['Código operación'])

costes_cardio_df = costes_df.set_index('Unnamed: 0')[operaciones_cardio]

x = lp.LpVariable.dicts("x", [(i, j) for i in operaciones_cardio for j in quirofanos], cat='Binary')

# Restricción 1:
for i in operaciones_cardio:
    problema += lp.lpSum(x[(i, j)] for j in quirofanos) >= 1

# Restricción 2:
for i in operaciones_cardio:
    for h in incompatibilidades[i]:
        for j in quirofanos:
            problema += x[(i, j)] + x[(h, j)] <= 1

problema += lp.lpSum(costes_cardio_df.loc[j, i] * x[(i, j)] for i in operaciones_cardio for j in quirofanos)
problema.solve()

print("Estado de la solución:", lp.LpStatus[problema.status])
print("Coste total:", lp.value(problema.objective))

asignaciones = []
for i in operaciones_cardio:
    for j in quirofanos:
        if x[(i, j)].varValue == 1:
            asignaciones.append((i, j))

asignaciones_df = pd.DataFrame(asignaciones, columns=["Operación", "Quirófano"])
asignaciones_df.to_excel("asignaciones_problema_1.xlsx", index=False)
print("Asignaciones guardadas en 'asignaciones_problema_1.xlsx'")
