#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ENTREGA 3 - GRUPO K - GENERACION DE COLUMNAS

1º MIO - MCA 

David Rodríguez Moratilla
Juan Francisco García Quijano
Lucía Varona Vidaurrazaga
Lucía Campos Díaz
"""

import pandas as pd
import pulp as lp

#IMPORTAR DATOS
datos_operaciones = pd.read_excel("241204_datos_operaciones_programadas.xlsx")
datos_costes = pd.read_excel("241204_costes.xlsx")

print(datos_operaciones.columns)


#PLANIFICACIONES FACTIBLES

especialidades = ["Cardiología Pediátrica", "Cirugía Cardíaca Pediátrica", 
                  "Cirugía Cardiovascular", "Cirugía General y del Aparato Digestivo"]
operaciones = datos_operaciones[datos_operaciones["Especialidad quirúrgica"].isin(especialidades)].copy()

operaciones["Hora inicio "] = pd.to_datetime(operaciones["Hora inicio "])
operaciones["Hora fin"] = pd.to_datetime(operaciones["Hora fin"])


def operaciones_solapan(inicio1, fin1, inicio2, fin2):
    return inicio1<fin2 and inicio2<fin1


#Encontrar planificaciones sin que sea muy costoso:
# Generar planificaciones completas de manera iterativa (greedy)
def generar_planificaciones(operaciones, max_variaciones=2):
    planificaciones = []
    operaciones = operaciones.sort_values(by="Hora inicio ").reset_index(drop=True)
    operaciones_asignadas = set()

    for _, op in operaciones.iterrows():
        if op["Código operación"] in operaciones_asignadas:
            continue

        # Crear una nueva planificación
        planificacion = [op["Código operación"]]
        operaciones_asignadas.add(op["Código operación"])

        for _, op_candidata in operaciones.iterrows():
            if op_candidata["Código operación"] in operaciones_asignadas:
                continue

            # Verificar si la operación candidata es compatible con la planificación actual
            es_compatible = all(
                not operaciones_solapan(
                    operaciones.loc[operaciones["Código operación"] == op_existente, "Hora inicio "].iloc[0],
                    operaciones.loc[operaciones["Código operación"] == op_existente, "Hora fin"].iloc[0],
                    op_candidata["Hora inicio "],
                    op_candidata["Hora fin"]
                )
                for op_existente in planificacion
            )

            if es_compatible:
                planificacion.append(op_candidata["Código operación"])
                operaciones_asignadas.add(op_candidata["Código operación"])

        # Añadir la planificación completa
        planificaciones.append(planificacion)

        # Generar variaciones
        for _ in range(max_variaciones):
            if len(planificacion) > 1:
                variacion = planificacion.copy()
                variacion.pop()  # Eliminar una operación
                planificaciones.append(variacion)

    # Filtrar planificaciones redundantes
    planificaciones_unicas = []
    for plan in planificaciones:
        if plan not in planificaciones_unicas:
            planificaciones_unicas.append(plan)

    return planificaciones_unicas



# Generar planificaciones con heurística greedy
planificaciones_factibles = generar_planificaciones(operaciones)

# Calcular Bik y Ck
Bik = {}
Ck = []

for k, planificacion in enumerate(planificaciones_factibles, 0):
    coste_planificacion = 0
    for op in planificacion:
        # Coste medio de la operación
        coste_operacion = datos_costes[op].mean()
        Bik[(op, k)] = 1
        coste_planificacion += coste_operacion
    Ck.append(coste_planificacion)

K = range(len(planificaciones_factibles))


#MODELO
problema = lp.LpProblem("Modelo2", lp.LpMinimize)

y = lp.LpVariable.dicts("y", [k for k in K], lowBound=0, upBound=1, cat=lp.LpBinary)


problema += lp.lpSum(Ck[k] * y[k] for k in K)


for op in operaciones["Código operación"]:
    problema += lp.lpSum(Bik.get((op, k), 0) * y[k] for k in K) >= 1


    
problema.solve()

lp.LpStatus[problema.status]


print("Objective = ", lp.value(problema.objective))


# Planificaciones seleccionadas
planificaciones_seleccionadas = [k for k in K if y[k].value() == 1]
print(f"Planificaciones seleccionadas: {planificaciones_seleccionadas}")

for k in planificaciones_seleccionadas:
    print(f"Planificación {k}: {planificaciones_factibles[k]}")





