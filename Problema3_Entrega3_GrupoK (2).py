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
import time


#IMPORTAR DATOS
operaciones = pd.read_excel("241204_datos_operaciones_programadas.xlsx")
operaciones.index = operaciones["Código operación"] #set up del index con el codigo de operación en la tabla operaciones
costes = pd.read_excel("241204_costes.xlsx")

#CONJUNTOS
# Extraigo los conjuntos (columnas de los distintos dataframes) y se convierten en listas para facilitar la posterior gestión 
quirofanosC = costes['Unnamed: 0'].tolist() #'Unnamed: 0' --> hace referencia a la columna de quirofanos 
operacionesC = operaciones.index.tolist()
equiposC = operaciones['Equipo de Cirugía'].tolist() #probablemente no sea necesario


#ASEGURAR TIPO DE DATO --> proceso de confirmación del tipo de dato de las columnas asociadas a tiempos
operaciones["Hora inicio "] = pd.to_datetime(operaciones["Hora inicio "])
operaciones["Hora fin"] = pd.to_datetime(operaciones["Hora fin"])


#%%

#FUNCION SOLAPAN: identificar si operaciones se solapan
def operaciones_solapan(inicio1, fin1, inicio2, fin2):
    return inicio1<fin2 and inicio2<fin1

#===========================================================================================================================

#FUNCIÓN GENERACIÓN CONJUNTO INICIAL MODELO 2: Generar planificaciones completas de manera iterativa (greedy, visto en Complejidad y redes)
def generar_planificaciones(operaciones):
    planificaciones = []
    operaciones = operaciones.sort_values(by="Hora inicio ").reset_index(drop=True)
    operaciones_asignadas = set()

    for _, op in operaciones.iterrows():
        if op["Código operación"] in operaciones_asignadas:
            continue

        # Crear una nueva planificación
        planificacion = [op]
        operaciones_asignadas.add(op["Código operación"])

        for _, op_candidata in operaciones.iterrows():
            if op_candidata["Código operación"] in operaciones_asignadas:
                continue

            # Verificar si la operación candidata es compatible con la planificación actual
            es_factible = all(
                not operaciones_solapan(
                    op_existente["Hora inicio "], op_existente["Hora fin"],
                    op_candidata["Hora inicio "], op_candidata["Hora fin"]
                )
                for op_existente in planificacion
            )

            if es_factible:
                planificacion.append(op_candidata)
                operaciones_asignadas.add(op_candidata["Código operación"])

        # Añadir la planificación completa a la lista de planificaciones
        planificaciones.append([op["Código operación"] for op in planificacion])

    return planificaciones

#GUARDADO DE DATOS RELACIONADO AL CONJUNTO INICIAL MODELO 2
planificaciones_factibles = generar_planificaciones(operaciones)
df_planificaciones = pd.DataFrame(planificaciones_factibles).transpose()
nombres_columnas = [f'Q{i+1}' for i in range(df_planificaciones.shape[1])]
df_planificaciones.columns = nombres_columnas 


#SALVO EN EXCEL LA INFORMACIÓN PARA CLARIDAD DE REPORTE
# Crear DataFrame de planificaciones factibles esto lo hacemos para verificar que el conjunto de factibles es óptimo (análisis humano)
data_planificaciones = []

# Cada planificación en `planificaciones_factibles` se trata como un quirófano
for k, planificacion in enumerate(planificaciones_factibles):
    for op in planificacion:
        # Añadimos el quirófano, la operación, la hora de inicio y la hora de fin
        data_planificaciones.append({
            "Quirófano": f"Q{k+1}",
            "Código operación": op,
            "Hora inicio": operaciones.loc[op]["Hora inicio "],
            "Hora fin":  operaciones.loc[op]["Hora fin"]
        })

# Crear un DataFrame con las planificaciones factibles, incluyendo las horas de inicio y fin
df_planificaciones_con_horas = pd.DataFrame(data_planificaciones)

# Guardar el DataFrame en un archivo Excel nuevo
df_planificaciones_con_horas.to_excel("planificaciones_factibles_con_horas.xlsx", index=False)

#===========================================================================================================================

#SEGUNDO CONJUNTO INCIAL: Creación de un segundo conjunto inicial de datos con una operación por quirófano
planificaciones_factibles_simplificadas =[]

for op in operacionesC:
     planificaciones_factibles_simplificadas.append([op])

#Por sencillez se obtiene el excel incluido en el reporte modificando el previo generado

#===========================================================================================================================

#FUNCIÓN ENCARGADA DE RESOLVER EL PROBLEMA MAESTRO RELAJADO
def resolver_modelo_maestro(planificaciones, operaciones):
    #INICIALIZO PARAMETRO Bik
    Bik = pd.DataFrame(0, index=operacionesC, columns=range(len(planificaciones)))
    for k, planificacion in enumerate(planificaciones, 0):  
        for op in planificacion:
            Bik.loc[(op, k)] = 1    
                
    #OBJETO PROBLEMA
    problema = lp.LpProblem("Modelo3", lp.LpMinimize)
    
    #VARIABLES    
    y = lp.LpVariable.dicts("y", [k for k in range(len(planificaciones))], lowBound=0, upBound=1, cat=lp.LpContinuous)
    
    #F.O
    problema += lp.lpSum(y[k] for k in range(len(planificaciones)))
    
    #RESTRICCIONES
    for op in operacionesC:
        problema += lp.lpSum(Bik.loc[(op, k)] * y[k] for k in range(len(planificaciones))) >= 1
        
    #RESOLUCION PROBLEMA
    problema.solve()
    
    return problema, y

#===========================================================================================================================

#FUNCIÓN ENCARGADA DE RESOLVER EL PROBLEMA MAESTRO SIN RELAJADO
def resolver_modelo_maestro_sinrelajar(planificaciones, operaciones):
    #INICIALIZO PARAMETRO Bik
    Bik = pd.DataFrame(0, index=operacionesC, columns=range(len(planificaciones)))
    for k, planificacion in enumerate(planificaciones, 0):  
        for op in planificacion:
            Bik.loc[(op, k)] = 1    
                
    #OBJETO PROBLEMA
    problema = lp.LpProblem("Modelo3", lp.LpMinimize)
    
    #VARIABLES    
    y = lp.LpVariable.dicts("y", [k for k in range(len(planificaciones))], lowBound=0, upBound=1, cat=lp.LpBinary)
    
    #F.O
    problema += lp.lpSum(y[k] for k in range(len(planificaciones)))
    
    #RESTRICCIONES
    for op in operacionesC:
        problema += lp.lpSum(Bik.loc[(op, k)] * y[k] for k in range(len(planificaciones))) >= 1
        
    #RESOLUCION PROBLEMA
    problema.solve()
    
    return problema, y

#===========================================================================================================================
#FUNCIÓN ENCARGADA DE RESOLVER EL SUBPROBLEMA 
def resolver_subproblema(precios_sombra, operaciones):
    #OBJETO PROBLEMA
    subproblema = lp.LpProblem("Generar_Nueva_Columna", lp.LpMaximize)
    
    #VARIABLES 
    x = lp.LpVariable.dicts("Operacion", [aux_x for aux_x in operacionesC], lowBound=0, upBound=1, cat=lp.LpBinary)
    
    # F.O
    subproblema += lp.lpSum(precios_sombra.loc[i] * x[i] for i in operacionesC)
    #RESTRICCIONES
    for i in operacionesC:
        for j in operacionesC:
            if i != j and operaciones_solapan(operaciones.loc[i]["Hora inicio "], operaciones.loc[i]["Hora fin"], operaciones.loc[j]["Hora inicio "], operaciones.loc[j]["Hora fin"]):
                subproblema += x[i] + x[j] <= 1
    
    subproblema.solve()
    
    # Generar nueva planificación a partir del resultado
    nueva_planificacion = [i for i in operacionesC if x[i].varValue is not None and int(x[i].varValue) == 1]
    return nueva_planificacion, lp.value(subproblema.objective)
    
    


#%%

#MAIN CONJUNTO INICIAL UNA PLANIFICACIÓN POR QUIROFANO
#algoritmo while partiendo de una planificación por quirofano

#Registrar tiempos 
inicio = time.time()
  
while True:
    #Resolver el modelo maestro
    modelo, y = resolver_modelo_maestro(planificaciones_factibles_simplificadas, operaciones)
    
    #Obtener precios sombra
    precios_sombra = [r.pi for r in modelo.constraints.values()]
    precios_sombra_df= pd.DataFrame(precios_sombra, index=operacionesC)
    
    #Resolver el subproblema
    nueva_planificacion, coste_reducido = resolver_subproblema( precios_sombra_df, operaciones)
    
    print("Nueva planificación generada:", nueva_planificacion)  # Depuración
    
    #Verificar condición de parada
    if coste_reducido <= 1:  
        break

    #Agregar la nueva planificación al modelo maestro
    if nueva_planificacion:
        planificaciones_factibles_simplificadas.append(nueva_planificacion)    
        
#Registro final
fin = time.time()
tiempo_transcurrido = fin - inicio
modelo_sinrelajar, y_sinrelajar= resolver_modelo_maestro_sinrelajar(planificaciones_factibles_simplificadas, operaciones)

#===========================================================================================================================
#EXPORTAR RESULTADOS

#QUIROFANOS UTILIZADOS
numero_minimo_quirófanosC1 = lp.value(modelo_sinrelajar.objective)

#Crear una lista para almacenar los datos de las planificaciones
planificaciones_dataC1 = []

#Recorremos las variables 'y' y sus valores
for k, var in y_sinrelajar.items():
    if var.varValue > 0:
        planificaciones_dataC1.append({"Número de Planificación": k,"Operaciones": ", ".join(planificaciones_factibles_simplificadas[k])})

#Crear un DataFrame de pandas para las planificaciones
df_planificacionesC1 = pd.DataFrame(planificaciones_dataC1)

#Crear un DataFrame para el resumen (número mínimo de quirófanos)
df_resumenC1 = pd.DataFrame({"Número mínimo de quirófanos": [numero_minimo_quirófanosC1]})

#Exportar a un archivo Excel
with pd.ExcelWriter("planificaciones_resultadosC1.xlsx") as writer:
    df_resumenC1.to_excel(writer, sheet_name="Resumen", index=False)
    df_planificacionesC1.to_excel(writer, sheet_name="Planificaciones", index=False)

#Mensaje de confirmación
print("Resultados exportados a 'planificaciones_resultadosC1.xlsx'")


#%%

#MAIN CONJUNTO INICIAL 2
#algoritmo while partiendo de la planificación del modelo2

#Registrar tiempos 
inicio2 = time.time()

while True:
    #Resolver el modelo maestro
    modelo, y = resolver_modelo_maestro(planificaciones_factibles, operaciones)
    
    #Obtener precios sombra
    precios_sombra = [r.pi for r in modelo.constraints.values()]
    precios_sombra_df= pd.DataFrame(precios_sombra, index=operacionesC)
    
    #Resolver el subproblema
    nueva_planificacion, coste_reducido = resolver_subproblema( precios_sombra_df, operaciones)
    
    print("Nueva planificación generada:", nueva_planificacion)  # Depuración
    
    #Verificar condición de parada
    if coste_reducido <= 1:  
        break
    

    #Agregar la nueva planificación al modelo maestro
    if nueva_planificacion:
        planificaciones_factibles.append(nueva_planificacion)    

#Registro final
fin2 = time.time()
tiempo_transcurrido2 = fin2 - inicio2
modelo_sinrelajar, y_sinrelajar= resolver_modelo_maestro_sinrelajar(planificaciones_factibles, operaciones)
#===========================================================================================================================
#EXPORTAR RESULTADOS

#QUIROFANOS UTILIZADOS
numero_minimo_quirófanosC2 = lp.value(modelo.objective)

# Crear una lista para almacenar los datos de las planificaciones
planificaciones_dataC2 = []

# Recorremos las variables 'y' y sus valores
for k, var in y.items():
    if var.varValue > 0:
        planificaciones_dataC2.append({"Número de Planificación": k,"Operaciones": ", ".join(planificaciones_factibles[k])})

#Crear un DataFrame de pandas para las planificaciones
df_planificacionesC2 = pd.DataFrame(planificaciones_dataC2)

#Crear un DataFrame para el resumen (número mínimo de quirófanos)
df_resumenC2 = pd.DataFrame({"Número mínimo de quirófanos": [numero_minimo_quirófanosC2]})

#Exportar a un archivo Excel
with pd.ExcelWriter("planificaciones_resultadosC2.xlsx") as writer:
    df_resumenC2.to_excel(writer, sheet_name="Resumen", index=False)
    df_planificacionesC2.to_excel(writer, sheet_name="Planificaciones", index=False)

#Mensaje de confirmación
print("Resultados exportados a 'planificaciones_resultadosC2.xlsx'")
