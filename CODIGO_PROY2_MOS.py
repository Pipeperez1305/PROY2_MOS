Importación de Librerías Necesarias
python
Copiar código
import pulp
import pandas as pd
import numpy as np
import requests
import csv
import plotly.graph_objects as go
Definición de Datos y Parámetros
Conjuntos y Parámetros Iniciales
python
Copiar código
# Conjuntos
N = []  # Conjunto de nodos (clientes, depósitos y nodos de recarga)
C = []  # Conjunto de clientes
D = []  # Conjunto de depósitos
R = []  # Conjunto de nodos de recarga
V = []  # Conjunto de vehículos

# Diccionarios para almacenar datos
node_coordinates = {}  # Coordenadas de los nodos
demanda_i = {}         # Demanda de cada cliente
s_i = {}               # Tiempo de servicio en cada cliente
C_v = {}               # Capacidad de carga de cada vehículo
R_v = {}               # Rango operativo de cada vehículo
C_km_v = {}            # Costo por kilómetro de cada vehículo
C_min_v = {}           # Costo por minuto de cada vehículo
F_mantenimiento_v = {} # Costo de mantenimiento diario de cada vehículo
C_fuel_v = {}          # Costo de recarga/combustible de cada vehículo
t_recarga_v = {}       # Tiempo de recarga/combustible de cada vehículo
Eficiencia_combustible_v = {}  # Eficiencia de combustible (Gas Car)
Eficiencia_energetica_v = {}   # Eficiencia energética (Solar EV)
consumo_combustible_v = {}     # Consumo de combustible por km (Gas Car)
consumo_energia_v = {}         # Consumo de energía por km (Solar EV)
S_d = {}               # Capacidad máxima de cada depósito
E_v = {}               # Capacidad máxima de energía de cada vehículo
Lectura de Datos desde Archivos
python
Copiar código
# Leer datos de nodos desde un archivo CSV (nodes.csv)
node_data = pd.read_csv('nodes.csv')  # Archivo que contiene ID de nodos, coordenadas y tipo
for index, row in node_data.iterrows():
    node_id = row['Node_ID']
    lat = row['Latitude']
    lon = row['Longitude']
    node_coordinates[node_id] = (lat, lon)
    N.append(node_id)
    if row['Type'] == 'Client':
        C.append(node_id)
        demanda_i[node_id] = row['Demand']
        s_i[node_id] = row['Service_Time']
    elif row['Type'] == 'Depot':
        D.append(node_id)
        S_d[node_id] = row['Capacity']
    elif row['Type'] == 'Recharge':
        R.append(node_id)

# Leer datos de vehículos desde un archivo CSV (vehicles.csv)
vehicle_data = pd.read_csv('vehicles.csv')  # Archivo con datos de vehículos
for index, row in vehicle_data.iterrows():
    vehicle_id = row['Vehicle_ID']
    V.append(vehicle_id)
    C_v[vehicle_id] = row['Capacity']
    R_v[vehicle_id] = row['Range']
    C_km_v[vehicle_id] = row['Freight_Rate']
    C_min_v[vehicle_id] = row['Time_Rate']
    F_mantenimiento_v[vehicle_id] = row['Daily_Maintenance']
    C_fuel_v[vehicle_id] = row['Fuel_Cost']
    t_recarga_v[vehicle_id] = row['Recharge_Time']
    if row['Type'] == 'Gas Car':
        Eficiencia_combustible_v[vehicle_id] = row['Gas_Efficiency']
        consumo_combustible_v[vehicle_id] = 1 / Eficiencia_combustible_v[vehicle_id]
    elif row['Type'] == 'Solar EV':
        Eficiencia_energetica_v[vehicle_id] = row['Electricity_Efficiency']
        consumo_energia_v[vehicle_id] = Eficiencia_energetica_v[vehicle_id]
    E_v[vehicle_id] = R_v[vehicle_id]  # Asumimos que la capacidad de energía es igual al rango para simplificar
Cálculo de Distancias y Tiempos Usando OSRM
python
Copiar código
def get_osrm_matrix(node_coords):
    # Construir la URL de la solicitud
    coordinates = ';'.join([f"{lon},{lat}" for lat, lon in node_coords.values()])
    url = f"http://router.project-osrm.org/table/v1/driving/{coordinates}?annotations=distance,duration"
    response = requests.get(url)
    data = response.json()

    if data['code'] == 'Ok':
        distances = data['distances']
        durations = data['durations']
        return distances, durations
    else:
        print("Error al obtener la matriz de OSRM")
        return None, None

# Obtener las matrices
distances_matrix, durations_matrix = get_osrm_matrix(node_coordinates)

# Mapear las matrices a los diccionarios d_ij y t_ij
d_ij = {}
t_ij = {}
nodes = list(node_coordinates.keys())
for idx_i, i in enumerate(nodes):
    for idx_j, j in enumerate(nodes):
        if i != j:
            d_ij[i,j] = distances_matrix[idx_i][idx_j] / 1000  # Convertir de metros a kilómetros
            t_ij[i,j] = durations_matrix[idx_i][idx_j] / 60    # Convertir de segundos a minutos
Definición del Problema de Optimización
python
Copiar código
# Definir el problema de optimización
prob = pulp.LpProblem("Vehicle_Routing_Problem", pulp.LpMinimize)
Variables de Decisión
Variables Principales
python
Copiar código
# Variables x_{ijv}: 1 si el vehículo v viaja de i a j
x = {}
for v in V:
    for i in N:
        for j in N:
            if i != j:
                x[i,j,v] = pulp.LpVariable(name=f"x_{i}_{j}_{v}", cat='Binary')
Variables de Carga y Energía
python
Copiar código
# Variables de carga u_{iv}: carga acumulada del vehículo v al llegar al nodo i
u = {}
for v in V:
    for i in N:
        u[i,v] = pulp.LpVariable(name=f"u_{i}_{v}", lowBound=0, upBound=C_v[v], cat='Continuous')

# Variables de energía e_{iv}: energía restante del vehículo v al llegar al nodo i
e = {}
for v in V:
    for i in N:
        e[i,v] = pulp.LpVariable(name=f"e_{i}_{v}", lowBound=0, upBound=E_v[v], cat='Continuous')

# Variables z_{iv}: 1 si el vehículo v recarga en el nodo i
z = {}
for v in V:
    for i in R:
        z[i,v] = pulp.LpVariable(name=f"z_{i}_{v}", cat='Binary')
#Variables de Asignación de Vehículos a Depósitos
python
Copiar código
# Variables y_{vd}: 1 si el vehículo v está asignado al depósito d
y = {}
for v in V:
    for d in D:
        y[v,d] = pulp.LpVariable(name=f"y_{v}_{d}", cat='Binary')
Función Objetivo
python
Copiar código
# Costo por distancia recorrida
cost_distance = pulp.lpSum(C_km_v[v] * d_ij[i,j] * x[i,j,v]
                           for v in V for i in N for j in N if i != j)

# Cálculo del tiempo total de operación por vehículo
T_v = {}
for v in V:
    travel_time = pulp.lpSum(t_ij[i,j] * x[i,j,v]
                             for i in N for j in N if i != j)
    service_time = pulp.lpSum(s_i.get(i, 0) * pulp.lpSum(x[i,j,v]
                             for j in N if i != j) for i in N)
    recharge_time = t_recarga_v[v] * 10 * pulp.lpSum(z[i,v] for i in R)  # Multiplicamos por 10 para el 100% de recarga
    T_v[v] = travel_time + service_time + recharge_time

# Costo de tiempo de operación
cost_time = pulp.lpSum(C_min_v[v] * T_v[v] for v in V)

# Costo de combustible/energía
cost_fuel = 0
if consumo_combustible_v:
    cost_fuel += pulp.lpSum(
        C_fuel_v[v] * consumo_combustible_v[v] * d_ij[i,j] * x[i,j,v]
        for v in V if v in consumo_combustible_v
        for i in N for j in N if i != j
    )
if consumo_energia_v:
    cost_fuel += pulp.lpSum(
        C_fuel_v[v] * consumo_energia_v[v] * d_ij[i,j] * x[i,j,v]
        for v in V if v in consumo_energia_v
        for i in N for j in N if i != j
    )

# Costo de carga (proceso de carga de productos)
C_carga_kg = 100  # Costo por kilogramo (COP/kg)
cost_carga = pulp.lpSum(
    C_carga_kg * demanda_i.get(i, 0) * pulp.lpSum(x[i,j,v] for j in N if i != j)
    for v in V for i in N
)

# Costo de mantenimiento diario
cost_maintenance = pulp.lpSum(F_mantenimiento_v[v] for v in V)

# Costo de recarga (combustible/energía)
cost_recharge = pulp.lpSum(
    C_fuel_v[v] * (E_v[v] - e[i,v]) * z[i,v]
    for v in V for i in R if v in C_fuel_v
)

# Función objetivo total
prob += cost_distance + cost_time + cost_fuel + cost_carga + cost_maintenance + cost_recharge
Restricciones
1. Asignación de Vehículos a Depósitos
python
Copiar código
# Cada vehículo está asignado a exactamente un depósito
for v in V:
    prob += pulp.lpSum(y[v,d] for d in D) == 1

# Los vehículos solo pueden salir y regresar al depósito al que están asignados
for v in V:
    for d in D:
        prob += pulp.lpSum(x[d,j,v] for j in N if d != j) == y[v,d]
        prob += pulp.lpSum(x[i,d,v] for i in N if i != d) == y[v,d]

# Evitar que los vehículos visiten otros depósitos distintos al asignado
for v in V:
    for d1 in D:
        for d2 in D:
            if d1 != d2:
                prob += pulp.lpSum(x[d2,j,v] for j in N if d2 != j) == 0
                prob += pulp.lpSum(x[i,d2,v] for i in N if i != d2) == 0
2. Cada Cliente es Atendido Exactamente una Vez
python
Copiar código
for i in C:
    prob += pulp.lpSum(x[i,j,v] for v in V for j in N if i != j) == 1
3. Conservación de Flujo
python
Copiar código
for v in V:
    for i in N:
        prob += (pulp.lpSum(x[i,j,v] for j in N if i != j) - pulp.lpSum(x[j,i,v] for j in N if i != j)) == 0
4. Restricciones de Capacidad de Carga
python
Copiar código
for v in V:
    for i in N:
        for j in N:
            if i != j:
                prob += u[j,v] >= u[i,v] + demanda_i.get(j, 0) - C_v[v] * (1 - x[i,j,v])

# Capacidad máxima y mínima
for v in V:
    for i in N:
        prob += u[i,v] >= demanda_i.get(i, 0)
        prob += u[i,v] <= C_v[v]
5. Restricciones de Energía y Recarga
python
Copiar código
# Consumo de energía al moverse de i a j
for v in V:
    for i in N:
        for j in N:
            if i != j:
                consumo = 0
                if v in consumo_energia_v:
                    consumo = consumo_energia_v[v] * d_ij[i,j]
                elif v in consumo_combustible_v:
                    consumo = consumo_combustible_v[v] * d_ij[i,j]
                prob += e[j,v] >= e[i,v] - consumo + E_v[v] * (x[i,j,v] - 1)

# Nivel inicial de energía en el depósito
for v in V:
    for d in D:
        prob += e[d,v] == E_v[v] * y[v,d]

# Recarga en nodos de recarga
for v in V:
    for i in R:
        prob += e[i,v] <= E_v[v] * z[i,v]
        prob += e[i,v] >= E_v[v] * z[i,v]

# Capacidad de energía
for v in V:
    for i in N:
        prob += e[i,v] >= 0
        prob += e[i,v] <= E_v[v]
6. Restricciones de Rango Operativo
python
Copiar código
# Asegurar que la distancia total recorrida entre recargas no exceda el rango
for v in V:
    prob += pulp.lpSum(d_ij[i,j] * x[i,j,v] for i in N for j in N if i != j) <= R_v[v]
7. Capacidad de los Depósitos
python
Copiar código
for d in D:
    prob += pulp.lpSum(demanda_i.get(i, 0) * x[d,i,v] for v in V for i in N if i != d) <= S_d[d]
Resolución del Problema
python
Copiar código
# Resolver el problema utilizando el solver HiGHS
prob.solve(pulp.HighsSolver())

# Mostrar el estado de la solución
print("Estado de la solución:", pulp.LpStatus[prob.status])

# Obtener el valor óptimo de la función objetivo
print("Costo total:", pulp.value(prob.objective))
Extracción y Almacenamiento de las Rutas
python
Copiar código
# Extraer las rutas y almacenarlas
vehicle_routes = {}

for v in V:
    route = []
    current_node = None
    for d in D:
        if pulp.value(y[v,d]) == 1:
            current_node = d
            break
    if current_node is None:
        continue
    route.append(current_node)
    while True:
        next_node = None
        for j in N:
            if current_node != j and pulp.value(x[current_node, j, v]) == 1:
                next_node = j
                break
        if next_node is None or next_node == current_node:
            break
        route.append(next_node)
        current_node = next_node
        if current_node in D:
            break
    # Guardar la ruta y sus coordenadas
    route_coords = [(node_coordinates[node][0], node_coordinates[node][1]) for node in route]
    vehicle_routes[v] = {
        'nodes': route,
        'coords': route_coords
    }

# Guardar las rutas en un archivo CSV
with open('ruta.csv', 'w', newline='') as csvfile:
    fieldnames = ['ID-Vehiculo', 'ID-Origen', 'ID-Destino']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    for v, data in vehicle_routes.items():
        nodes = data['nodes']
        for idx in range(len(nodes)-1):
            writer.writerow({'ID-Vehiculo': v,
                             'ID-Origen': nodes[idx],
                             'ID-Destino': nodes[idx+1]})
Informe de Costos Operacionales
python
Copiar código
# Cálculo de costos individuales
print("Costo por distancia recorrida:", pulp.value(cost_distance))
print("Costo de tiempo de operación:", pulp.value(cost_time))
print("Costo de combustible/energía:", pulp.value(cost_fuel))
print("Costo de carga:", pulp.value(cost_carga))
print("Costo de mantenimiento:", pulp.value(cost_maintenance))
print("Costo de recarga:", pulp.value(cost_recharge))
Visualización de Rutas
python
Copiar código
# Visualización de rutas
fig = go.Figure()

# Añadir las rutas de cada vehículo al mapa
for v, data in vehicle_routes.items():
    lats = [coord[0] for coord in data['coords']]
    lons = [coord[1] for coord in data['coords']]
    fig.add_trace(go.Scattermapbox(
        lat=lats,
        lon=lons,
        mode='markers+lines',
        name=f'Vehículo {v}',
        marker=go.scattermapbox.Marker(size=9),
        text=data['nodes'],
        hoverinfo='text'
    ))

# Configurar el diseño del mapa
fig.update_layout(
    mapbox=dict(
        style='open-street-map',
        zoom=10,
        center=dict(lat=np.mean([coord[0] for coords in vehicle_routes.values() for coord in coords['coords']]),
                    lon=np.mean([coord[1] for coords in vehicle_routes.values() for coord in coords['coords']]))
    ),
    margin={"r":0,"t":0,"l":0,"b":0}
)

# Mostrar el mapa
fig.show()