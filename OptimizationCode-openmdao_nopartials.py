# -*- coding: utf-8 -*-
"""
Created on Sat Apr 15 05:58:03 2023

@author: ardya

To ensure that this code runs smoothly, please make sure to install the necessary dependencies. These include pandas, numpy, openmdao, cartopy, and matplotlib. You can install these dependencies using pip, by running the following command in your terminal:

pip install pandas numpy openmdao cartopy matplotlib

Once you have installed these packages, you should be all set to run the code without any issues.

"""
import pandas as pd
import numpy as np
import openmdao.api as om
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt

# Assuming 1 week of operations
week = 1
days = 7 * week

# Airport_info
Airport_info_df = pd.read_excel(
    'Exercise_Interview_Data_modified.xlsx', sheet_name='Airport_info', index_col=0)
Airport_info = Airport_info_df.values

# Distances
Distances_df = pd.read_excel(
    'Exercise_Interview_Data_modified.xlsx', sheet_name='Distances', index_col=0)
Distances_df.reset_index(drop=False, inplace=True)
Distances_df.drop(Distances_df.columns[0], axis=1, inplace=True)
Distances = Distances_df.values
num_distances = len(Distances)

# Demand
Demand_df = pd.read_excel('Exercise_Interview_Data_modified.xlsx',
                          sheet_name='Demand', index_col=0)
Demand_df.reset_index(drop=False, inplace=True)
Demand_df.drop(Demand_df.columns[0], axis=1, inplace=True)
Demand = Demand_df.values
Demand = Demand * days # assuming Demand is per day
num_demand = len(Demand)

# Aircraft_data
Aircraft_data_df = pd.read_excel(
    'Exercise_Interview_Data_modified.xlsx', sheet_name='Aircraft_data', index_col=0)
Aircraft_data = Aircraft_data_df.values
num_aircraft_data = len(Aircraft_data[0])

# Constants

BT = 10 * days * week  # hours per day
LF = 0.8
LF_15 = np.zeros(shape=((num_aircraft_data, num_distances, num_distances)))
for k in range(num_aircraft_data):
    LF_15[k, :, :] = LF
G = Aircraft_data[10, :]
R = Aircraft_data[4, :]
e = 0.07
f = 1.42
CL = Aircraft_data[6, :]
V = Aircraft_data[0, :]
V_15 = np.zeros(shape=((num_aircraft_data, num_distances, num_distances)))
for k in range(num_aircraft_data):
    V_15[k, :, :] = V[k]
    
CX = Aircraft_data[7, :]
CX_15 = np.zeros(shape=((num_aircraft_data, num_distances, num_distances)))
for k in range(num_aircraft_data):
    CX_15[k, :, :] = CX[k]
    
s = Aircraft_data[1, :]
s_15 = np.zeros(shape=((num_aircraft_data, num_distances, num_distances)))
for k in range(num_aircraft_data):
    s_15[k, :, :] = s[k]

TAT = np.zeros(num_aircraft_data)
for k in range(num_aircraft_data):
    if G[k] != 0:
        TAT[k] = (Aircraft_data[2, k] + Aircraft_data[3, k]) / 2 / 60
    else:
        TAT[k] = (Aircraft_data[2, k] + Aircraft_data[3, k]) / 60
TAT_15 = np.zeros(shape=((num_aircraft_data, num_distances, num_distances)))
for k in range(num_aircraft_data):
    TAT_15[k, :, :] = TAT[k]
    
g = np.zeros(num_distances)
for h in range(num_distances):
    if h == 0:
        g[h] = 0
    else:
        g[h] = 1

hub_check = np.zeros_like(Distances)
for i in range(num_distances):
    for j in range(num_distances):
        if i == 0 or j == 0:
            hub_check[i, j] = 0.7
        else:
            hub_check[i, j] = 1
hub_check_15 = np.zeros(shape=((num_aircraft_data, num_distances, num_distances)))
for i in range(num_distances):
    for j in range(num_distances):
        hub_check_15[:, i, j] = hub_check[i,j]
        for k in range(num_aircraft_data):
            np.fill_diagonal(hub_check_15[k], 0)
            
AC_runway = Aircraft_data[5, :]
runway_check = np.random.randint(0, 1, size=((num_aircraft_data, num_distances, num_distances)))
for k in range(num_aircraft_data):
    for i in range(num_distances):
        for j in range(num_distances):
            if AC_runway[k] <= Airport_info[i, 3] and AC_runway[k] <= Airport_info[j, 3]:
                runway_check[k, i, j] = 1
            else:
                runway_check[k, i, j] = 0

distance_check = np.random.randint(0, 1, size=((num_aircraft_data, num_distances, num_distances)))
for k in range(num_aircraft_data):
    for i in range(num_distances):
        for j in range(num_distances):
            if 2 * Distances[i, j] <= R[k] and G[k] != 0 and i != j and (i == 0 or j == 0):
                distance_check[k, i, j] = 1
            elif Distances[i, j] <= R[k] and G[k] == 0 and i != j:
                distance_check[k, i, j] = 1
            else:
                distance_check[k, i, j] = 0
                
CT = np.zeros(shape=(num_aircraft_data, num_distances, num_distances))
for k in range(num_aircraft_data):
    for i in range(num_distances):
        for j in range(num_distances):
            CT[k, i, j] = CL[k] * Distances[i, j] / V[k]

cF = Aircraft_data[9, :]
CF = np.zeros(shape=(num_aircraft_data, num_distances, num_distances))
for k in range(num_aircraft_data):
    for i in range(num_distances):
        for j in range(num_distances):
            CF[k, i, j] = cF[k] * f * Distances[i, j] / 1.5

CE = np.zeros(shape=(num_aircraft_data, num_distances, num_distances))
for k in range(num_aircraft_data):
    for i in range(num_distances):
        for j in range(num_distances):
            CE[k, i, j] = e * G[k] * Distances[i, j] / R[k]

Y_eur = np.zeros(shape=(num_distances, num_distances))
for i in range(num_distances):
    for j in range(num_distances):
        if Distances[i, j] == 0:
            Y_eur[i, j] = 0
        else:
            Y_eur[i, j] = 5.9 * (Distances[i, j] ** -0.76) + 0.043


x = np.random.randint(0, 219, size=(num_distances, num_distances))
np.fill_diagonal(x, 0)


w = np.random.randint(0, 219, size=(num_distances, num_distances))
np.fill_diagonal(w, 0)

z = np.random.randint(0, 10, size=((num_aircraft_data, num_distances, num_distances)))
z = z * runway_check * distance_check

n_AC = np.random.randint(0, 2, size=((num_aircraft_data)))
n_AC_15 = np.zeros(shape=((num_aircraft_data, num_distances, num_distances)))
for k in range(num_aircraft_data):
    n_AC_15[k, :, :] = n_AC[k]

class DemandConstraint(om.ExplicitComponent):
    def setup(self):
        self.add_input('x', shape=((num_distances, num_distances)))
        self.add_input('w', shape=((num_distances, num_distances)))
        
        self.add_output('trf_check', shape=((num_distances, num_distances)))
        self.add_output('pax_check', shape=((num_distances, num_distances)))
        
        rows = np.arange(num_distances * num_distances)
        cols = np.arange(num_distances * num_distances)
        self.declare_partials('trf_check', 'w', method='fd', rows=rows, cols=cols)

        self.declare_partials('pax_check', 'x', method='fd', rows=rows, cols=cols)
        self.declare_partials('pax_check', 'w', method='fd', rows=rows, cols=cols)
        
    def compute(self, inputs, outputs):
        x = inputs['x']
        for k in range(num_aircraft_data):
            for i in range(num_distances):
                for j in range(num_distances):
                    if z.sum(axis=0)[i,j] == 0:
                        x[i,j] = 0
                    np.fill_diagonal(x, 0)
                    
        w = inputs['w']
        for k in range(num_aircraft_data):
            for i in range(num_distances):
                for j in range(num_distances):
                    if z.sum(axis=0)[i,j] == 0:
                        w[i,j] = 0
                    np.fill_diagonal(w, 0)
        # Demand Constraint
        trf_check = np.zeros((num_distances, num_distances))
        for i in range(num_distances):
            for j in range(num_distances):
                if i == j or i == 0 or j == 0:
                    trf_check[i, j] = 0
                else:
                    trf_check[i, j] = w[i, j] / (Demand[i, j] * g[i] * g[j])
        outputs['trf_check'] = trf_check
        
        pax_check = np.zeros((num_distances, num_distances))
        for i in range(num_distances):
            for j in range(num_distances):
                if i == j:
                    pax_check[i,j] = 0
                else:
                    pax_check[i,j] = (x[i,j] + w[i,j]) / Demand[i,j]
        outputs['pax_check'] = pax_check
                    
class CapacityConstraint(om.ExplicitComponent):
    def setup(self):
        self.add_input('x', shape=((num_distances, num_distances)))
        self.add_input('w', shape=((num_distances, num_distances)))
        self.add_input('z', shape=((num_aircraft_data, num_distances, num_distances)))
        
        self.add_output('capacity_check', shape=((num_distances, num_distances)))
        
        rows = np.arange(num_distances * num_distances)
        cols = np.arange(num_distances * num_distances)
        self.declare_partials('capacity_check', 'x', method='fd', rows=rows, cols=cols)
        self.declare_partials('capacity_check', 'w', method='fd', rows=rows, cols=cols)
        self.declare_partials('capacity_check', 'z', method='fd')

    def compute(self, inputs, outputs):
        z = inputs['z']
        z = z * runway_check * distance_check
        for k in range(num_aircraft_data):
            for i in range(num_distances):
                for j in range(num_distances):
                    z[k,i,j] = z[k,j,i]
                    np.fill_diagonal(z[k], 0)
                    if n_AC[k] == 0:
                        z[k] = 0
        
        x = inputs['x']
        for k in range(num_aircraft_data):
            for i in range(num_distances):
                for j in range(num_distances):
                    if z.sum(axis=0)[i,j] == 0:
                        x[i,j] = 0
                    np.fill_diagonal(x, 0)
                    
        w = inputs['w']
        for k in range(num_aircraft_data):
            for i in range(num_distances):
                for j in range(num_distances):
                    if z.sum(axis=0)[i,j] == 0:
                        w[i,j] = 0
                    np.fill_diagonal(w, 0)

        # Capacity Constraint
        capacity_check = np.zeros((num_distances, num_distances))
        for i in range(num_distances):
            for j in range(num_distances):
                for k in range(num_aircraft_data):
                    for m in range(num_distances):
                        if i == j or i == 0 or j == 0:
                            capacity_check[i, j] = 0
                        else:
                            capacity_check[i, j] =  (x[i, j] + w.sum(axis=1)[i] * (1 - g[j]) + w.sum(axis=0)[j] * (1 - g[i])) / np.sum(z * s_15 * LF_15, axis=0)[i,j]
        non_finite_indices = ~np.isfinite(capacity_check)
        capacity_check[non_finite_indices] = 0
        outputs['capacity_check'] = capacity_check
    
        
class UtilizationConstraint(om.ExplicitComponent):
    def setup(self):
        self.add_input('z', shape=((num_aircraft_data, num_distances, num_distances)))
        self.add_input('n_AC', shape=((num_aircraft_data)))

        self.add_output('utilization_check', shape=((num_aircraft_data)))
        
        rows = np.arange(num_aircraft_data)
        cols = np.arange(num_aircraft_data)
        
        self.declare_partials('utilization_check', 'z', method='fd')
        self.declare_partials('utilization_check', 'n_AC', method='fd', rows=rows, cols=cols)

    def compute(self, inputs, outputs):
        z = inputs['z']
        z = z * runway_check * distance_check

        n_AC = inputs['n_AC']
        n_AC_15 = np.zeros(shape=((num_aircraft_data, num_distances, num_distances)))
        for k in range(num_aircraft_data):
            n_AC_15[k, :, :] = n_AC[k]
            
        # Utilization Constraint
        utilization_check = np.zeros((num_aircraft_data))
        for k in range(num_aircraft_data):
            for i in range(num_distances):
                for j in range(num_distances):
                    utilization_check[k] = np.sum((Distances / V_15 + TAT_15) * z, axis=(1,2))[k] / (BT * n_AC[k])
        non_finite_indices = ~np.isfinite(utilization_check)
        utilization_check[non_finite_indices] = 0 
                
        outputs['utilization_check'] = utilization_check
        

class Profit(om.ExplicitComponent):
    def setup(self):
        self.add_input('x', shape=((num_distances, num_distances)))
        self.add_input('w', shape=((num_distances, num_distances)))
        self.add_input('z', shape=((num_aircraft_data, num_distances, num_distances)))
        self.add_input('n_AC', shape=((num_aircraft_data)))
        
        self.add_output('profit')
        
        self.declare_partials('profit', 'x', method='fd')
        self.declare_partials('profit', 'w', method='fd')
        self.declare_partials('profit', 'z', method='fd')
        self.declare_partials('profit', 'n_AC', method='fd')
        
    def compute(self, inputs, outputs):
        n_AC = inputs['n_AC']
        
        z = inputs['z']
        z = z * runway_check * distance_check
        for k in range(num_aircraft_data):
            for i in range(num_distances):
                for j in range(num_distances):
                    z[k,i,j] = z[k,j,i]
                    np.fill_diagonal(z[k], 0)
                    if n_AC[k] == 0:
                        z[k] = 0
        
        x = inputs['x']
        for k in range(num_aircraft_data):
            for i in range(num_distances):
                for j in range(num_distances):
                    if z.sum(axis=0)[i,j] == 0:
                        x[i,j] = 0
                    np.fill_diagonal(x, 0)
                    
        w = inputs['w']
        for k in range(num_aircraft_data):
            for i in range(num_distances):
                for j in range(num_distances):
                    if z.sum(axis=0)[i,j] == 0:
                        w[i,j] = 0
                    np.fill_diagonal(w, 0)

        # Objective Function
        
        total_profit = np.sum(Y_eur * Distances * (x + 0.9*w) - np.sum((hub_check_15 * (CX_15 + CT + CF) + CE) * z, axis=0)) - np.sum(n_AC * CL * week)
        outputs['profit'] = total_profit
        
prob = om.Problem()

prob.model.add_subsystem('demand_constraint', DemandConstraint(), promotes=['*'])
prob.model.add_subsystem('capacity_constraint', CapacityConstraint(), promotes=['*'])
prob.model.add_subsystem('utilization_constraint', UtilizationConstraint(), promotes=['*'])
prob.model.add_subsystem('profit', Profit(), promotes=['*'])

prob.model.linear_solver = om.LinearBlockGS()

prob.driver = om.ScipyOptimizeDriver()
prob.driver.options['optimizer'] = 'SLSQP'

prob.model.add_design_var('x', lower=0)
prob.model.add_design_var('w', lower=0)
prob.model.add_design_var('z', lower=0)
prob.model.add_design_var('n_AC', lower=0)

prob.model.add_objective('profit', scaler=-1)

prob.model.add_constraint('trf_check', lower=0, upper=1)
prob.model.add_constraint('pax_check', lower=0, upper=1)
prob.model.add_constraint('capacity_check', lower=0, upper=1)
prob.model.add_constraint('utilization_check', lower=0, upper=1)

prob.setup(check=True, derivatives=True)
prob.run_model()
# prob.check_partials(compact_print=True)
# prob.check_totals(compact_print=True)
prob.run_driver()

# Print results
n_AC_opt = prob.get_val('n_AC')
z_opt = prob.get_val('z')
x_opt = prob.get_val('x')
w_opt = prob.get_val('w')
profit_opt = prob.get_val('profit')
utilization_opt = prob.get_val('utilization_check')

linewidth = z_opt.sum(axis=0)

# Latitude & Longitude
AP_lat = {k: v['Latitude (deg)'] for k, v in Airport_info_df.items()}
AP_lon = {k: v['Longitude (deg)'] for k, v in Airport_info_df.items()}

# Set bounding box to Italy
extent = [6.6, 18.5, 35.3, 47.1]

# Create map
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
ax.set_extent(extent, crs=ccrs.PlateCarree())

# Add map features
ax.add_feature(cfeature.LAND.with_scale('50m'))
ax.add_feature(cfeature.OCEAN.with_scale('50m'))
ax.add_feature(cfeature.COASTLINE.with_scale('50m'))
ax.add_feature(cfeature.BORDERS.with_scale('50m'))

# Plot cities
for ICAO_code in AP_lat:
    ax.scatter(AP_lon[ICAO_code], AP_lat[ICAO_code], marker='.',
               s=100, color='blue', transform=ccrs.PlateCarree())
    ax.text(AP_lon[ICAO_code], AP_lat[ICAO_code], ICAO_code,
            fontsize=15, transform=ccrs.PlateCarree())

# Set hub airport
hub = 'LIMC'

# Add a marker for the hub location
ax.plot(AP_lon[hub], AP_lat[hub], 'o', markersize=10, color='red')

# Plot lines between cities
for i in Airport_info_df.keys():
    start = Airport_info_df[i]
    for j in Airport_info_df.keys():
        end = Airport_info_df[j]
        if start == end:
            continue
        else:
            ax.plot([start['Longitude (deg)'], end['Longitude (deg)']],
                    [start['Latitude (deg)'], end['Latitude (deg)']], 
                    color='blue', linewidth=linewidth[i,j], marker='.', markersize=1,
                    transform=ccrs.Geodetic())