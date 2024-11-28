import pulp
import pandas as pd
import requests


class VehicleRoutingModel:
    """Clase para modelar y resolver problemas de ruteo vehicular con múltiples casos."""

    def __init__(self):
        """Inicializa los atributos principales del modelo."""
        self.prob = None  # Problema de optimización
        self.N = []  # Conjunto de nodos (clientes, depósitos, recarga)
        self.C = []  # Conjunto de clientes
        self.D = []  # Conjunto de depósitos
        self.R = []  # Conjunto de nodos de recarga
        self.V = []  # Conjunto de vehículos
        self.parameters = {}  # Parámetros comunes y específicos por nodo o vehículo
        self.case_number = None  # Número del caso actual
        self.distances_matrix = None  # Matriz de distancias entre nodos
        self.durations_matrix = None  # Matriz de tiempos entre nodos
        self.x = {}  # Variables de decisión (viajes)
        self.u = {}  # Variables de carga
        self.e = {}  # Variables de energía
        self.z = {}  # Variables de recarga
        self.y = {}  # Variables de asignación de vehículos


    def load_case_data(self, case_number):
        """Carga los datos de entrada desde archivos específicos según el caso."""
        self.case_number = case_number
        base_path = f"./case_{case_number}/"
        self.node_data = pd.read_csv(f"{base_path}Clients.csv")
        self.vehicle_data = pd.read_csv(f"{base_path}Vehicles.csv")
        self.depot_data = pd.read_csv(f"{base_path}Depots.csv")
        if case_number in [4, 5, 6]:
            self.depot_capacity = pd.read_csv(f"{base_path}DepotCapacities.csv")
        if case_number == 6:
            self.recharge_nodes = pd.read_csv(f"{base_path}RechargeNodes.csv")
        self.initialize_parameters()

    def initialize_parameters(self): #REVISAR
        """Inicializa parámetros comunes y específicos para nodos y vehículos."""
        # Configuración para nodos
        for _, row in self.node_data.iterrows():
            node_id = row['Node_ID']
            self.N.append(node_id)
            if row['Type'] == 'Client':
                self.C.append(node_id)
                self.parameters[node_id] = {
                    'demand': row['Demand'],
                    'service_time': row['Service_Time']
                }
            elif row['Type'] == 'Depot':
                self.D.append(node_id)
                self.parameters[node_id] = {'capacity': row['Capacity']}
            elif row['Type'] == 'Recharge' and self.case_number == 6:
                self.R.append(node_id)

        # Configuración para vehículos
        for _, row in self.vehicle_data.iterrows():
            vehicle_id = row['Vehicle_ID']
            self.V.append(vehicle_id)
            self.parameters[vehicle_id] = {
                'capacity': row['Capacity'],
                'range': row['Range'],
                'freight_rate': row['Freight_Rate'],
                'time_rate': row['Time_Rate'],
                'daily_maintenance': row['Daily_Maintenance'],
                'fuel_cost': row['Fuel_Cost'],
                'recharge_time': row['Recharge_Time'],
                'type': row['Type'],
                'efficiency': row.get('Gas_Efficiency', row.get('Electricity_Efficiency')),
            }
            if row['Type'] == 'Gas Car':
                self.parameters[vehicle_id]['fuel_consumption'] = 1 / row['Gas_Efficiency']
            elif row['Type'] == 'Solar EV':
                self.parameters[vehicle_id]['energy_consumption'] = row['Electricity_Efficiency']

        # Calcular matriz de distancias y tiempos
        self.distances_matrix, self.durations_matrix = self.get_osrm_matrix()
        #self.configure_case()

    def get_osrm_matrix(self):
        """Obtiene las matrices de distancias y tiempos entre nodos usando OSRM."""
        coordinates = ';'.join([f"{lon},{lat}" for lat, lon in self.node_data[['Latitude', 'Longitude']].values])
        url = f"http://router.project-osrm.org/table/v1/driving/{coordinates}?annotations=distance,duration"
        response = requests.get(url)
        data = response.json()
        if data['code'] == 'Ok':
            distances = data['distances']
            durations = data['durations']
            d_ij = {}
            t_ij = {}
            nodes = self.node_data['Node_ID'].tolist()
            for idx_i, i in enumerate(nodes):
                for idx_j, j in enumerate(nodes):
                    if i != j:
                        d_ij[i, j] = distances[idx_i][idx_j] / 1000  # Convertir de metros a kilómetros
                        t_ij[i, j] = durations[idx_i][idx_j] / 60    # Convertir de segundos a minutos
            return d_ij, t_ij
        else:
            raise Exception("Error al obtener la matriz de OSRM")
        
    def define_decision_variables(self):
        """Define las variables de decisión del modelo."""
        for v in self.V:
            for i in self.N:
                for j in self.N:
                    if i != j:
                        self.x[i, j, v] = pulp.LpVariable(name=f"x_{i}_{j}_{v}", cat='Binary')

        for v in self.V:
            for i in self.N:
                self.u[i, v] = pulp.LpVariable(name=f"u_{i}_{v}", lowBound=0, upBound=self.parameters[v]['capacity'], cat='Continuous')
                self.e[i, v] = pulp.LpVariable(name=f"e_{i}_{v}", lowBound=0, upBound=self.parameters[v]['range'], cat='Continuous')

        for v in self.V:
            for i in self.R:
                self.z[i, v] = pulp.LpVariable(name=f"z_{i}_{v}", cat='Binary')

        for v in self.V:
            for d in self.D:
                self.y[v, d] = pulp.LpVariable(name=f"y_{v}_{d}", cat='Binary')

    def define_objective(self):
        """Define la función objetivo del modelo."""
        cost_distance = pulp.lpSum(
            self.parameters[v]['freight_rate'] * self.distances_matrix[i, j] * self.x[i, j, v]
            for v in self.V for i in self.N for j in self.N if i != j
        )
        cost_time = pulp.lpSum(
            self.parameters[v]['time_rate'] * self.durations_matrix[i, j] * self.x[i, j, v]
            for v in self.V for i in self.N for j in self.N if i != j
        )
        cost_maintenance = pulp.lpSum(
            self.parameters[v]['daily_maintenance'] for v in self.V
        )
        cost_fuel = pulp.lpSum(
            self.parameters[v].get('fuel_cost', 0) * self.parameters[v].get('efficiency', 0) * self.distances_matrix[i, j] * self.x[i, j, v]
            for v in self.V for i in self.N for j in self.N if i != j
        )
        self.prob += cost_distance + cost_time + cost_maintenance + cost_fuel

    def define_general_constraints(self):
        """Define las restricciones generales aplicables a todos los casos."""
        # Asignación de vehículos a depósitos
        for v in self.V:
            self.prob += pulp.lpSum(self.y[v, d] for d in self.D) == 1

        # Conservación de flujo
        for v in self.V:
            for i in self.N:
                self.prob += (pulp.lpSum(self.x[i, j, v] for j in self.N if i != j) -
                              pulp.lpSum(self.x[j, i, v] for j in self.N if i != j)) == 0

    def configure_case(self):
        """Configura restricciones y parámetros específicos según el caso."""
        if self.case_number == 1:
            self.setup_standard_case()
        elif self.case_number == 2:
            self.setup_increment_vehicles_case()
        elif self.case_number == 3:
            self.setup_long_distance_case()
        elif self.case_number == 4:
            self.setup_limited_capacity_case()
        elif self.case_number == 5:
            self.setup_multiple_products_case()
        elif self.case_number == 6:
            self.setup_recharge_nodes_case()

    def setup_standard_case(self):
        """Configura restricciones para el caso estándar."""
        print("Configurando caso estándar...")
        """
        Configura restricciones para el caso estándar.
        Escenario Base: Operación Estándar.
        Por cada vehículo existente, hay exactamente 2 clientes que requieren abastecimiento.
        La distancia promedio entre los clientes y los centros de distribución es de 5 km (±1 km).
        La demanda de cada cliente varía entre 8 kg y 20 kg.
        Objetivo: Minimizar costos operativos y tiempo de entrega.
        """

        
        # Restricción 1: Cada vehículo debe atender exactamente 2 clientes.
        for v in self.V:
            self.prob += pulp.lpSum(self.x[i, j, v] for i in self.C for j in self.C if i != j) == 2, f"Atender_exactamente_2_clientes_{v}"

        # Restricción 2: Cada cliente debe ser atendido exactamente una vez.
        for i in self.C:
            self.prob += pulp.lpSum(self.x[i, j, v] for v in self.V for j in self.N if i != j) == 1, f"Atender_cliente_una_vez_{i}"

        # Restricción 3: Los vehículos solo pueden partir de un depósito asignado.
        for v in self.V:
            for d in self.D:
                self.prob += pulp.lpSum(self.x[d, j, v] for j in self.N if d != j) == self.y[v, d], f"Partida_desde_deposito_{v}_{d}"

        # Restricción 4: Los vehículos deben regresar al depósito asignado.
        for v in self.V:
            for d in self.D:
                self.prob += pulp.lpSum(self.x[i, d, v] for i in self.N if i != d) == self.y[v, d], f"Regreso_a_deposito_{v}_{d}"

        # Restricción 5: Conservación de flujo para cada vehículo.
        for v in self.V:
            for i in self.N:
                self.prob += (
                    pulp.lpSum(self.x[i, j, v] for j in self.N if i != j) -
                    pulp.lpSum(self.x[j, i, v] for j in self.N if i != j)
                ) == 0, f"Conservacion_de_flujo_{v}_{i}"

        # Restricción 6: Capacidad de los vehículos.
        for v in self.V:
            for i in self.C:
                self.prob += (
                    pulp.lpSum(self.u[j, v] for j in self.C if j != i) >= self.parameters[i]['demand']
                ), f"Capacidad_minima_{v}_{i}"
                self.prob += (
                    self.u[i, v] <= self.parameters[v]['capacity']
                ), f"Capacidad_maxima_{v}_{i}"

        # Objetivo: Minimizar costos operativos y tiempo de entrega.
        self.define_objective()

    def setup_increment_vehicles_case(self):
        """Configura restricciones para el incremento de vehículos."""
        print("Configurando caso 2: Incremento de Vehículos...")
        # Restricción 1: Cada cliente debe ser atendido exactamente una vez.
        for i in self.C:
            self.prob += pulp.lpSum(self.x[i, j, v] for v in self.V for j in self.N if i != j) == 1, f"Atender_cliente_una_vez_{i}"

        # Restricción 2: Cada vehículo debe salir de un único depósito asignado.
        for v in self.V:
            for d in self.D:
                self.prob += pulp.lpSum(self.x[d, j, v] for j in self.N if d != j) == self.y[v, d], f"Partida_desde_deposito_{v}_{d}"

        # Restricción 3: Cada vehículo debe regresar al depósito asignado.
        for v in self.V:
            for d in self.D:
                self.prob += pulp.lpSum(self.x[i, d, v] for i in self.N if i != d) == self.y[v, d], f"Regreso_a_deposito_{v}_{d}"

        # Restricción 4: Conservación de flujo para cada vehículo.
        for v in self.V:
            for i in self.N:
                self.prob += (
                    pulp.lpSum(self.x[i, j, v] for j in self.N if i != j) -
                    pulp.lpSum(self.x[j, i, v] for j in self.N if i != j)
                ) == 0, f"Conservacion_de_flujo_{v}_{i}"

        # Restricción 5: Capacidad de carga de los vehículos.
        for v in self.V:
            for i in self.C:
                self.prob += (
                    pulp.lpSum(self.u[j, v] for j in self.C if j != i) >= self.parameters[i]['demand']
                ), f"Capacidad_minima_{v}_{i}"
                self.prob += (
                    self.u[i, v] <= self.parameters[v]['capacity']
                ), f"Capacidad_maxima_{v}_{i}"

        # Restricción 6: Los recursos adicionales deben ser usados de manera eficiente.
        # La proporción de vehículos a clientes está garantizada en los datos cargados,
        # por lo que aquí solo aseguramos que los vehículos sean utilizados óptimamente.
        for v in self.V:
            self.prob += pulp.lpSum(self.x[i, j, v] for i in self.C for j in self.C if i != j) >= 1, f"Uso_optimo_vehiculo_{v}"

        # Objetivo: Minimizar los costos operativos considerando la flota ampliada.
        self.define_objective()
        
    def setup_long_distance_case(self):
        """
        DESCRIPCIÓN:
        Escenario 3: Distancias Largas.
        - La distancia promedio entre centros de distribución y clientes se incrementa
        a 10 km, con una desviación estándar de 0.5 km.
        - La demanda por cliente se reduce para reflejar un escenario de menor carga por cliente.

        Objetivo:
        Evaluar la capacidad del modelo para manejar eficientemente rutas de larga distancia
        y analizar el impacto en los costos operativos debido al incremento de distancia en las entregas.
        """
        print("Configurando caso 3: Distancias Largas...")

        # Restricción 1: Cada cliente debe ser atendido exactamente una vez.
        for i in self.C:
            self.prob += pulp.lpSum(self.x[i, j, v] for v in self.V for j in self.N if i != j) == 1, f"Atender_cliente_una_vez_{i}"

        # Restricción 2: Cada vehículo debe partir de un único depósito asignado.
        for v in self.V:
            for d in self.D:
                self.prob += pulp.lpSum(self.x[d, j, v] for j in self.N if d != j) == self.y[v, d], f"Partida_desde_deposito_{v}_{d}"

        # Restricción 3: Cada vehículo debe regresar al depósito asignado.
        for v in self.V:
            for d in self.D:
                self.prob += pulp.lpSum(self.x[i, d, v] for i in self.N if i != d) == self.y[v, d], f"Regreso_a_deposito_{v}_{d}"

        # Restricción 4: Conservación de flujo para cada vehículo.
        for v in self.V:
            for i in self.N:
                self.prob += (
                    pulp.lpSum(self.x[i, j, v] for j in self.N if i != j) -
                    pulp.lpSum(self.x[j, i, v] for j in self.N if i != j)
                ) == 0, f"Conservacion_de_flujo_{v}_{i}"

        # Restricción 5: Capacidad de carga ajustada por menor demanda.
        for v in self.V:
            for i in self.C:
                self.prob += (
                    pulp.lpSum(self.u[j, v] for j in self.C if j != i) >= self.parameters[i]['demand']
                ), f"Capacidad_minima_{v}_{i}"
                self.prob += (
                    self.u[i, v] <= self.parameters[v]['capacity']
                ), f"Capacidad_maxima_{v}_{i}"

        # Restricción 6: Energía o combustible suficiente para largas distancias.
        for v in self.V:
            for i in self.N:
                for j in self.N:
                    if i != j:
                        consumo = 0
                        if self.parameters[v]['type'] == 'Gas Car':
                            consumo = self.parameters[v]['efficiency'] * self.distances_matrix[i, j]
                        elif self.parameters[v]['type'] == 'Solar EV':
                            consumo = self.parameters[v]['efficiency'] * self.distances_matrix[i, j]
                        self.prob += (
                            self.e[j, v] >= self.e[i, v] - consumo
                        ), f"Energia_suficiente_{v}_{i}_{j}"

        # Objetivo: Minimizar costos operativos ajustados para largas distancias.
        self.define_objective()

    def setup_limited_capacity_case(self):
        """

        DESCRIPCIÓN:
        Escenario 4: Centros de Distribución con Capacidad Limitada.
        - Cada centro de distribución tiene un límite máximo para el volumen de productos que puede distribuir,
        definido exclusivamente en el archivo `DepotCapacities.csv`.

        Objetivo:
        Adaptar el modelo para cumplir con restricciones de capacidad en los centros de distribución,
        evaluando el impacto de esta limitación en la planificación de las rutas y en la eficiencia general
        de las entregas.
        """
        print("Configurando caso 4: Capacidad Limitada en Depósitos...")

        # Leer capacidades de los depósitos desde el archivo DepotCapacities.csv
        depot_capacities = pd.read_csv('DepotCapacities.csv')
        depot_capacity_dict = {row['Depot_ID']: row['Max_Capacity'] for _, row in depot_capacities.iterrows()}

        # Restricción 1: Cada cliente debe ser atendido exactamente una vez.
        for i in self.C:
            self.prob += pulp.lpSum(self.x[i, j, v] for v in self.V for j in self.N if i != j) == 1, f"Atender_cliente_una_vez_{i}"

        # Restricción 2: Cada vehículo debe partir de un único depósito asignado.
        for v in self.V:
            for d in self.D:
                self.prob += pulp.lpSum(self.x[d, j, v] for j in self.N if d != j) == self.y[v, d], f"Partida_desde_deposito_{v}_{d}"

        # Restricción 3: Cada vehículo debe regresar al depósito asignado.
        for v in self.V:
            for d in self.D:
                self.prob += pulp.lpSum(self.x[i, d, v] for i in self.N if i != d) == self.y[v, d], f"Regreso_a_deposito_{v}_{d}"

        # Restricción 4: Conservación de flujo para cada vehículo.
        for v in self.V:
            for i in self.N:
                self.prob += (
                    pulp.lpSum(self.x[i, j, v] for j in self.N if i != j) -
                    pulp.lpSum(self.x[j, i, v] for j in self.N if i != j)
                ) == 0, f"Conservacion_de_flujo_{v}_{i}"

        # Restricción 5: Capacidad de carga de los vehículos.
        for v in self.V:
            for i in self.C:
                self.prob += (
                    pulp.lpSum(self.u[j, v] for j in self.C if j != i) >= self.parameters[i]['demand']
                ), f"Capacidad_minima_{v}_{i}"
                self.prob += (
                    self.u[i, v] <= self.parameters[v]['capacity']
                ), f"Capacidad_maxima_{v}_{i}"

        # Restricción 6: Capacidad máxima de los depósitos.
        for d in self.D:
            max_capacity = depot_capacity_dict.get(d, float('inf'))  # Por si algún depósito no tiene capacidad definida
            self.prob += pulp.lpSum(
                self.parameters[i]['demand'] * self.x[d, i, v]
                for v in self.V for i in self.C if i != d
            ) <= max_capacity, f"Capacidad_maxima_deposito_{d}"

        # Objetivo: Minimizar costos operativos ajustados para la capacidad limitada.
        self.define_objective()


    def setup_multiple_products_case(self):
        """
        Configura restricciones para múltiples productos.

        DESCRIPCIÓN:
        Caso Especial 1: Múltiples Productos.
        - Cada cliente tiene demandas específicas para varios tipos de productos.
        - La capacidad de carga de los vehículos permanece constante.
        - Los vehículos deben transportar distintos tipos de productos en cada ruta.

        Objetivo:
        Evaluar la capacidad del modelo para manejar múltiples tipos de demanda y
        optimizar las rutas considerando las restricciones de capacidad de los vehículos con cargas mixtas.
        """
        print("Configurando caso 5: Múltiples Productos...")

        # Leer demandas de los clientes desde el archivo Clients.csv
        client_data = pd.read_csv('Clients.csv')  # Contiene demanda por tipo de producto
        product_types = [col for col in client_data.columns if col.startswith("Product_")]  # Identificar los productos
        client_demands = {
            row['Client_ID']: {ptype: row[ptype] for ptype in product_types}
            for _, row in client_data.iterrows()
        }

        # Leer capacidades de los depósitos desde el archivo DepotCapacities.csv
        depot_data = pd.read_csv('DepotCapacities.csv')  # Contiene capacidad por tipo de producto
        depot_capacities = {
            row['Depot_ID']: {ptype: row[ptype] for ptype in product_types}
            for _, row in depot_data.iterrows()
        }

        # Restricción 1: Cada cliente debe ser atendido exactamente una vez.
        for i in self.C:
            self.prob += pulp.lpSum(self.x[i, j, v] for v in self.V for j in self.N if i != j) == 1, f"Atender_cliente_una_vez_{i}"

        # Restricción 2: Cada vehículo debe partir de un único depósito asignado.
        for v in self.V:
            for d in self.D:
                self.prob += pulp.lpSum(self.x[d, j, v] for j in self.N if d != j) == self.y[v, d], f"Partida_desde_deposito_{v}_{d}"

        # Restricción 3: Cada vehículo debe regresar al depósito asignado.
        for v in self.V:
            for d in self.D:
                self.prob += pulp.lpSum(self.x[i, d, v] for i in self.N if i != d) == self.y[v, d], f"Regreso_a_deposito_{v}_{d}"

        # Restricción 4: Conservación de flujo para cada vehículo.
        for v in self.V:
            for i in self.N:
                self.prob += (
                    pulp.lpSum(self.x[i, j, v] for j in self.N if i != j) -
                    pulp.lpSum(self.x[j, i, v] for j in self.N if i != j)
                ) == 0, f"Conservacion_de_flujo_{v}_{i}"

        # Restricción 5: Capacidad de carga por tipo de producto.
        for v in self.V:
            for ptype in product_types:
                self.prob += pulp.lpSum(
                    client_demands[i][ptype] * self.x[i, j, v]
                    for i in self.C for j in self.N if i != j
                ) <= self.parameters[v]['capacity'], f"Capacidad_maxima_{v}_{ptype}"

        # Restricción 6: Capacidad máxima de los depósitos por tipo de producto.
        for d in self.D:
            for ptype in product_types:
                self.prob += pulp.lpSum(
                    client_demands[i][ptype] * self.x[d, i, v]
                    for v in self.V for i in self.C if i != d
                ) <= depot_capacities[d][ptype], f"Capacidad_maxima_deposito_{d}_{ptype}"

        # Objetivo: Minimizar costos operativos ajustados para cargas mixtas.
        self.define_objective()


    def setup_recharge_nodes_case(self):
        """
        Configura restricciones para nodos de recarga.

        DESCRIPCIÓN:
        Caso Especial 2: Nodos de Recarga.
        - Las distancias entre centros de distribución y clientes son grandes.
        - La proporción de vehículos a clientes es de 1:20.
        - El modelo debe optimizar el uso de nodos de recarga en rutas de larga distancia.

        Objetivo:
        Garantizar que el modelo pueda adaptarse y aprovechar los nodos de recarga,
        permitiendo una operación continua en rutas de larga distancia con una alta proporción
        de clientes por vehículo.
        """
        print("Configurando caso 6: Nodos de Recarga...")

        # Leer demandas de los clientes desde el archivo Clients.csv
        client_data = pd.read_csv('Clients.csv')  # Contiene demanda por tipo de producto
        product_types = [col for col in client_data.columns if col.startswith("Product_")]
        client_demands = {
            row['Client_ID']: {ptype: row[ptype] for ptype in product_types}
            for _, row in client_data.iterrows()
        }

        # Leer capacidades de los depósitos desde el archivo DepotCapacities.csv
        depot_data = pd.read_csv('DepotCapacities.csv')  # Contiene capacidad por tipo de producto
        depot_capacities = {
            row['Depot_ID']: {ptype: row[ptype] for ptype in product_types}
            for _, row in depot_data.iterrows()
        }

        # Leer nodos de recarga desde el archivo RechargeNodes.csv
        recharge_nodes = pd.read_csv('RechargeNodes.csv')
        self.R = recharge_nodes['Node_ID'].tolist()

        # Restricción 1: Cada cliente debe ser atendido exactamente una vez.
        for i in self.C:
            self.prob += pulp.lpSum(self.x[i, j, v] for v in self.V for j in self.N if i != j) == 1, f"Atender_cliente_una_vez_{i}"

        # Restricción 2: Cada vehículo debe partir de un único depósito asignado.
        for v in self.V:
            for d in self.D:
                self.prob += pulp.lpSum(self.x[d, j, v] for j in self.N if d != j) == self.y[v, d], f"Partida_desde_deposito_{v}_{d}"

        # Restricción 3: Cada vehículo debe regresar al depósito asignado.
        for v in self.V:
            for d in self.D:
                self.prob += pulp.lpSum(self.x[i, d, v] for i in self.N if i != d) == self.y[v, d], f"Regreso_a_deposito_{v}_{d}"

        # Restricción 4: Conservación de flujo para cada vehículo.
        for v in self.V:
            for i in self.N:
                self.prob += (
                    pulp.lpSum(self.x[i, j, v] for j in self.N if i != j) -
                    pulp.lpSum(self.x[j, i, v] for j in self.N if i != j)
                ) == 0, f"Conservacion_de_flujo_{v}_{i}"

        # Restricción 5: Capacidad de carga por tipo de producto.
        for v in self.V:
            for ptype in product_types:
                self.prob += pulp.lpSum(
                    client_demands[i][ptype] * self.x[i, j, v]
                    for i in self.C for j in self.N if i != j
                ) <= self.parameters[v]['capacity'], f"Capacidad_maxima_{v}_{ptype}"

        # Restricción 6: Capacidad máxima de los depósitos por tipo de producto.
        for d in self.D:
            for ptype in product_types:
                self.prob += pulp.lpSum(
                    client_demands[i][ptype] * self.x[d, i, v]
                    for v in self.V for i in self.C if i != d
                ) <= depot_capacities[d][ptype], f"Capacidad_maxima_deposito_{d}_{ptype}"

        # Restricción 7: Energía suficiente para recorrer largas distancias.
        for v in self.V:
            for i in self.N:
                for j in self.N:
                    if i != j:
                        consumo = 0
                        if self.parameters[v]['type'] == 'Gas Car':
                            consumo = self.parameters[v]['efficiency'] * self.distances_matrix[i, j]
                        elif self.parameters[v]['type'] == 'Solar EV':
                            consumo = self.parameters[v]['efficiency'] * self.distances_matrix[i, j]
                        self.prob += (
                            self.e[j, v] >= self.e[i, v] - consumo + self.parameters[v]['range'] * (1 - self.x[i, j, v])
                        ), f"Energia_suficiente_{v}_{i}_{j}"

        # Restricción 8: Permitir recarga en nodos de recarga.
        for v in self.V:
            for i in self.R:
                self.prob += self.e[i, v] <= self.parameters[v]['range'] * self.z[i, v], f"Recarga_maxima_{v}_{i}"
                self.prob += self.e[i, v] >= 0, f"Energia_minima_nodo_recarga_{v}_{i}"

        # Restricción 9: Priorización del nodo de recarga más cercano.
        for v in self.V:
            for i in self.R:
                for j in self.R:
                    if i != j:
                        self.prob += (
                            self.z[i, v] * self.distances_matrix[i, j] <=
                            self.z[j, v] * self.distances_matrix[j, j]
                        ), f"Prioridad_recarga_cercana_{v}_{i}_{j}"

        # Objetivo: Minimizar costos operativos incluyendo nodos de recarga.
        recharge_cost = pulp.lpSum(
            self.z[i, v] * self.parameters[v]['fuel_cost']  # Costo por recarga
            for v in self.V for i in self.R
        )
        total_cost = recharge_cost + pulp.lpSum(
            self.z[i, v] * self.distances_matrix[i, j]
            for v in self.V for i in self.R for j in self.N if i != j
        )
        self.prob += total_cost, "Costo_total_operativo"


    
    def solve(self):
        """Resuelve el problema de optimización usando HiGHS."""
        self.prob = pulp.LpProblem("Vehicle_Routing_Problem", pulp.LpMinimize)
        self.define_objective()
        self.prob.solve(pulp.HighsSolver())
        print(f"Estado del caso {self.case_number}: {pulp.LpStatus[self.prob.status]}")

    def export_results(self):
        """Exporta los resultados en un archivo CSV."""
        # Implementar exportación según el formato deseado
        pass


# Ejecución de los casos
model = VehicleRoutingModel()
for case in range(1, 7):
    model.load_case_data(case_number=case)
    model.solve()
    model.export_results()
