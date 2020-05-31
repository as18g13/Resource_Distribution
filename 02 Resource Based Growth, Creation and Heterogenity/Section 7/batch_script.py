#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 15 09:32:42 2018

@author: as18g13
"""

#import Growth_Model_nov_2018_rev1  as grm1
import os 
import numpy as np

growth_type =0
Initial_resource_per_agent =0
Initial_resource_per_cell =0
Resource_added_per_timestep=300
cost_to_add_cell =5
cost_to_add_agent =5
Optimum_resource =3
initial_coordinates=[50,50]
array_size=(100,100)
Initial_agent_number=1
time_steps=300
arrayx=100
arrayy=100
intial_previous_coordinates=(-9999, -9999)
initial_prev_agent=-9999
Initial_resource_per_agent_per_timestep=0
NS = 10
sim_num=0
SD=0
neighbour_num=0


for k in range(450,600,25):
    fname = 'global_parameters2.py'
    with open(fname, 'w') as f:
        f.write('growth_type ={}\n'.format(growth_type))
        f.write('Initial_resource_per_agent ={}\n'.format(Initial_resource_per_agent))
        f.write('Initial_resource_per_cell ={}\n'.format(Initial_resource_per_cell))
        f.write('Resource_added_per_timestep={}\n'.format(k))
        f.write('cost_to_add_cell ={}\n'.format(cost_to_add_cell))
        f.write('cost_to_add_agent ={}\n'.format(cost_to_add_cell))
        f.write('Optimum_resource ={}\n'.format(Optimum_resource))
        f.write('initial_coordinates={}\n'.format(initial_coordinates))
        f.write('array_size={}\n'.format((arrayx,arrayy)))
        f.write('Initial_agent_number={}\n'.format(Initial_agent_number))
        f.write('time_steps={}\n'.format(time_steps))
        f.write('arrayx={}\n'.format(arrayx))
        f.write('arrayy={}\n'.format(arrayy))
        f.write('intial_previous_coordinates={}\n'.format(intial_previous_coordinates))
        f.write('initial_prev_agent={}\n'.format(initial_prev_agent))
        f.write('Initial_resource_per_agent_per_timestep={}\n'.format(Initial_resource_per_agent_per_timestep))
        f.write('NS={}\n'.format(NS))
        f.write('sim_num={}\n'.format(sim_num))
        f.write('SD={}\n'.format(SD))
        f.write('neighbour_num={}\n'.format(neighbour_num))
    f.close()
    print (Resource_added_per_timestep,cost_to_add_cell,Optimum_resource,Initial_resource_per_agent,NS)
    os.system('python Section_7_Inequality_in_IIS.py')