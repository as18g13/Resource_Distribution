#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 15 09:32:42 2018

@author: as18g13
"""

import growth_model_rev2 as gmr2
import os 
import numpy as np

Resource_added_per_timestep = 400
cost_to_add_cell = 50
Optimum_resource = 5
Initial_resource_per_agent = 1

for i in np.arange(62,65,1):
    fname = 'global_parameters.py'
    with open(fname, 'w') as f:
        Initial_resource_per_agent=i
        f.write('Resource_added_per_timestep ={}\n'.format(Resource_added_per_timestep))
        f.write('cost_to_add_cell ={}\n'.format(cost_to_add_cell))
        f.write('Optimum_resource ={}\n'.format(Optimum_resource))
        f.write('Initial_resource_per_agent ={}'.format(Initial_resource_per_agent))
    f.close()
    print (Resource_added_per_timestep,cost_to_add_cell,Optimum_resource,Initial_resource_per_agent)
    os.system('python Section_6_Growth_Creation_Model.py')
