# -*- coding: utf-8 -*-
"""
Created on Thu Mar 22 11:27:58 2018

Growth and Distribution Model for a hetereogeneous multi-agent system



@author: as18g13

MIT License

Copyright (c) [2019] [Alexander John Howsam Stokes]

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

"""

from mpl_toolkits.axes_grid1 import make_axes_locatable
import os as os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as clr
from copy import deepcopy
import matplotlib.animation as manimation
import time
import matplotlib as mpl
from random import randint
import pandas as pd
import matplotlib.animation as animation
import csv
import math
import networkx as nx
import network_rev1 as net1
from global_parameters2 import growth_type,Initial_resource_per_agent, Initial_resource_per_cell,Resource_added_per_timestep,cost_to_add_cell,cost_to_add_agent, \
Optimum_resource,initial_coordinates,arrayx,arrayy,intial_previous_coordinates,initial_prev_agent,Initial_agent_number,time_steps,Initial_resource_per_agent_per_timestep,NS,sim_num,SD,neighbour_num

#######################Parameters##############################################
moore = [(-1, -1),(-1, 0),(-1, 1),(0, -1),(0, 1),(1, -1),(1,0),(1,1)] 
Full_moore = [(-1, -1),(-1, 0),(-1, 1),(0, -1),(0,0),(0, 1),(1, -1),(1,0),(1,1)] 
neumann = [(-1,0),(0,1),(1,0),(0,-1)]

#The parameters in greeen are set in the separate global parameters file
"""
growth_type =  0 #Growth type 0 - deterministic, 1 - locally stochastic, 2 - globally stochastic, 3 - global selection model, 4 - random combo model
Initial_resource_per_agent = 0
Initial_resource_per_agent_per_timestep = 0
Resource_added_per_timestep = 20
cost_to_add_cell = 5
cost_to_add_agent = 5
Optimum_resource = 10
initial_coordinates = [50,50]
arrayx=100
arrayy=100

intial_previous_coordinates = -9999,-9999
initial_prev_agent = -9999
Initial_agent_number = 1
time_steps = 200
cost_to_add_cell = 5
cost_to_add_agent = 5
Optimum_resource = 10
"""
neigh_pref_on='Y'
max_growth2 = 'N' #maximum growth per timestep
pref_growth1 = 'N' #Preferential growth only allowing 
agent_heterogeniety = 'Y'

area = np.zeros((arrayx,arrayy),dtype=int)
pass_down_list = []
agents = []
cells = []
allocated_cells = []
agent_lib = []
Multiple_starting_locations = 'N'
Num_simulations = 1
Resource_used_per_timestep = 1
min_limit = 0
Num_DCs = []
Num_RCs = []
total_resource_in_cells = []
total_res_in_system = []
ts_graph = []
function_time = {}
ginilib = []
min_agent_wealth = []
max_agent_wealth = []
pref_lib=[]
upstream_res=[]
downstream_res=[]

agent_data = []
wealth_data=[]
agent_upstream_data = []
agent_network = []
max_pass_per_timestep = 5

#########Defining Agents in system and resource cells###################################

class agent(): #Class which assigns agents to the system
    def __init__(self,POS,PREV,prev_agent,ID,ind_kin_group,neighbour,num,ext,timestamp,res,pass_,growth_count,\
                 ext_amount,ext_cells,ext_agents,cells,need_type,collect_need,pass_down,downstream_agents,ind_maint,\
                 ind_add_cell,ind_add_agent,death_count,net_need,received,upstream_res,up_down_change,ext_downstream,\
                 tot_downstream,ind_pref1,wealth,neighbourhood,neighbours,neigh_ave,neigh_max,neigh_pref,agent_num,upstream_agents):
        self.agent_ID=agent_num
        self.pos = POS #Position of agent in the system
        self.prev = PREV #Position of previous agent in system
        self.prev_agent=prev_agent
        self.id = ID #ID assign to agent
        self.ind_kin_group = ind_kin_group #Individual preference (kin or group selection)
        self.neighbours = neighbour #Neighbours of agent
        self.NH = num #Size of neighbourhood (1 for kin, greater for group)
        self.creation = timestamp
        self.extension = ext
        self.ext_amount = ext_amount
        self.ext_cells=ext_cells
        self.keep = res
        self.pass_ = pass_
        self.growth_count = growth_count
        self.cells = cells
        self.need_type = need_type
        self.collect_need = collect_need
        self.pass_down = pass_down
        self.downstream_agents = downstream_agents
        self.total_downstream_agents = tot_downstream
        self.ind_maint = ind_maint
        self.ind_add_cell = ind_add_cell
        self.ind_add_agent = ind_add_agent
        self.death_count = death_count
        self.net_need = net_need
        self.received = received
        self.upstream_res = upstream_res
        self.up_down_change = up_down_change
        self.ext_downstream = ext_downstream
        self.ind_preference = ind_pref1
        self.neighbourhood_pref = neigh_pref
        self.wealth = wealth
        self.neighbourhood_size = neighbourhood
        self.neighbours=neighbours
        self.neigh_ave=neigh_ave
        self.neigh_max = neigh_max
        self.upstream_agents=upstream_agents

class cell():
    def __init__(self,POS,agent_location,agent_,res,cost_paid):
        self.pos = POS 
        self.agent_loc = agent_location
        self.agent_=agent_
        self.resource = res
        self.cost_paid = cost_paid
            
####################################Agent Extend check algorithm#############################################     
def find_neighbours(agent): #function to find adajcent neighbour 
    if len(agent.neighbours)==agent.neighbourhood_size:
        pass
    else:
        x,y = agent.pos
        
        for i in range(len(agents)):
            ag=agents[np.random.randint(0,len(agents))]
            if ag==agent:
                pass
            elif ag in agent.neighbours:
                pass
            else:
                xx,yy=ag.pos
                if (xx,yy)==initial_coordinates:
                    pass
                elif xx>(x-NS) and xx<(x+NS) and yy>(y-NS) and yy<(y+NS):
                    if len(ag.neighbours)<ag.neighbourhood_size:
#                        if abs(ag.ind_preference-agent.ind_preference)<0.15:
                        agent.neighbours.append(ag)
                        ag.neighbours.append(agent)
                if len(agent.neighbours)==agent.neighbourhood_size:
                    break
    if neigh_pref_on=='Y':
        if agent.neighbours:
            total_neigh_pref=0
            for neigh in agent.neighbours:
                total_neigh_pref+=neigh.ind_preference
            agent.neighbourhood_pref=(total_neigh_pref/len(agent.neighbours))
            new_ind = (agent.neighbourhood_pref+agent.ind_preference)/2
            agent.ind_preference=np.random.normal(new_ind,SD,1)
    if agent.neighbours:
        neighbour_res = 0
        neighbour_max = 0
        for agg in agent.neighbours:
            neighbour_res+=agg.keep
            if agg.keep>neighbour_max:
                neighbour_max=agg.keep
        agent.neigh_ave=float(neighbour_res)/len(agent.neighbours)
        agent.neigh_max=neighbour_max
        if agent.neigh_ave==0:
            max_neighbour_cells=0
            for agg in agent.neighbours:
                if len(agg.cells)>max_neighbour_cells:
                    max_neighbour_cells=len(agg.cells)
            if max_neighbour_cells>len(agent.cells):
                wealth_need=1
            else:
                wealth_need=0
            return wealth_need
        else:
            wealth_need=float((agent.neigh_ave*agent.ind_preference))
            #wealth_need=neighbour_max*agent.ind_preference
            if wealth_need<agent.keep:
                return 0
            else:
                return wealth_need-agent.keep
    else:
        agent.ind_preference=np.random.normal(agent.ind_preference,SD,1)
        wealth_need=float((agent.keep*agent.ind_preference))
        if wealth_need<agent.keep:
            return 0
        else:
            return wealth_need-agent.keep
                                  
def pref_growth():
    max_ext = 0
    for ag in agents:
        if ag.ext_amount>max_ext:
            max_ext=ag.ext_amount
    return max_ext

def downstream_extension():
    for agent in agents:
        if agent.downstream_agents:
            agent.ext_downstream ='N'
    for agent in agents:
        if agent.pos==initial_coordinates:
            pass
        else:
            if not agent.downstream_agents and agent.extension=='Y':
                upstream_agent=agent.prev_agent
                while type(upstream_agent)!=int:
                    upstream_agent.ext_downstream = 'Y'
                    temp=upstream_agent
                    upstream_agent=temp.prev_agent
    return 0

def pass_to_keep_ind(agent): #Function which passes resource for the agent to keep and use from the passed down resource
    if agent.net_need>0:
        if agent.pass_>agent.net_need:
            agent.keep+=agent.net_need
            agent.pass_-=agent.net_need
        else:
            agent.keep+=agent.pass_
            agent.pass_=0
            agent.net_need-=agent.pass_
    agent.keep=round(agent.keep,5)
    agent.pass_=round(agent.pass_,5)
    return 0

def from_pass_to_keep():
    for agent in agents:
        pass_to_keep_ind(agent)
    return 0   

def pass_down_res():
    for agent in agents:
        up_ag = agent.prev_agent
        if type(up_ag)!=int:
            if len(up_ag.downstream_agents)>1:
                if agent.upstream_res>agent.received:
                    upstream_res.append(agent.upstream_res)
                    downstream_res.append(agent.received)
    return 0
    
def pass_function(next_upstream_agent): #function which distributes resource basesd on the needs downstream
    upstream_agent = next_upstream_agent[0]
    for aggs in upstream_agent.pass_down:
        next_upstream_agent.append(aggs)
    if upstream_agent.downstream_agents:
        resource_available1 = upstream_agent.pass_
        upstream_resource = deepcopy(resource_available1)
        resource_units = round(float(sum(upstream_agent.pass_down.values())),5)
        if resource_available1>resource_units:
            resource_available=resource_units
            upstream_agent.pass_-=resource_units
        else:
            resource_available=resource_available1
            upstream_agent.pass_=0
        if resource_available == 0 or resource_units==0:
            pass
        else:
            resource_proportion = round(resource_available/resource_units,5)
            for down_stream_agent in upstream_agent.downstream_agents:
                down_stream_collective_need = upstream_agent.pass_down[down_stream_agent]
                resource_allocation = round(down_stream_collective_need*resource_proportion,5)
                down_stream_agent.pass_+=resource_allocation   
                down_stream_agent.upstream_res = upstream_resource
                down_stream_agent.received=down_stream_agent.pass_
                if len(upstream_agent.downstream_agents)>1: 
                    down_stream_agent.up_down_change.append((down_stream_agent.upstream_res,down_stream_agent.received))
                pass_to_keep_ind(down_stream_agent)
                #next_upstream_agent.append(down_stream_agent)
        if max_growth2=='Y':
            downstream_extension()
            ext_dwn_agents = []
            for ag in upstream_agent.downstream_agents:
                if ag.ext_downstream == 'Ý' or ag.extension=='Y':
                    ext_dwn_agents.append(ag)
            if upstream_agent.pass_>0 and ext_dwn_agents:
                resource_available=upstream_agent.pass_
                resource_allocation = round(resource_available/len(ext_dwn_agents),5)
                upstream_agent.pass_=0
                for agg in upstream_agent.downstream_agents:
                    agg.pass_+=resource_allocation                
    del next_upstream_agent[0]
    return next_upstream_agent
                    
def pass_down_resource():
    next_upstream_agent = []
    next_upstream_agent.append(agents[0])
    pass_to_keep_ind(next_upstream_agent[0])
    while (next_upstream_agent):
        next_upstream_agent=pass_function(next_upstream_agent) 
    return 0        

def clear_pass_down():
    for agent in agents:
        agent.pass_down = {} 
        agent.cells_down = 0    

def culumative_need(agent,ind_need):
    upstream_agent = agent.prev_agent
    downstream_agent = agent
    while type(upstream_agent) != int:
        if downstream_agent in upstream_agent.pass_down.keys():
            upstream_agent.pass_down[downstream_agent]+=ind_need
            downstream_agent=upstream_agent
            upstream_agent=upstream_agent.prev_agent              
        else:
            upstream_agent.pass_down[downstream_agent]=ind_need
            downstream_agent=upstream_agent
            upstream_agent=upstream_agent.prev_agent               
    return 0            

def Calc_growth_need(): #Function which calculates the need of each agent by adding the cells which could be added as well as the resource required by 
    for agent in agents: #cells already present, and resource required by an additional agent extension
        agent.ind_add_agent = 0
        agent_extend_check(agent)
        if agent.ext_amount>0:
            agent.ind_add_agent+=(((cost_to_add_cell+Optimum_resource)*agent.ext_amount)+cost_to_add_agent)
    return 0

def Calc_add_cells_need():
    for agent in agents:
        agent.ind_add_cell = 0
        for cell in agent.cells:
            if cell.cost_paid == 'N':
                agent.ind_add_cell+=cost_to_add_cell+Optimum_resource    
    return 0

def Calc_maint_need():
    for agent in agents:
        agent.ind_maint = 0
        for cell in agent.cells:
            if cell.cost_paid == 'Y':
                if cell.resource<Optimum_resource:
                    agent.ind_maint+=(Optimum_resource-cell.resource)                
    return 0

def select_need_type():
    if pref_growth1=='Y':
        max_ext=pref_growth()
    else:
        max_ext=0
    for agent in agents:
        agent.net_need = 0
        if agent.ind_maint>0:
            agent.need_type='maint'
            if (agent.ind_maint-agent.keep)>0:
                agent.net_need = (agent.ind_maint-agent.keep)
            else:
                agent.net_need=0
        elif agent.ind_add_cell>0:
            agent.need_type='add_cell'
            if (agent.ind_add_cell-agent.keep)>0:
                agent.net_need = (agent.ind_add_cell-agent.keep)
            else:
                agent.net_need=0
        elif agent.ind_add_agent>0:
            agent.need_type='add_agent'
            if (agent.ind_add_agent-agent.keep)>0:
                agent.net_need = (agent.ind_add_agent-agent.keep)
            else:
                agent.net_need=0
        #else:
        if agent_heterogeniety=='Y':
            if agent.pos==initial_coordinates:
                pass
            else:
    #            agent.need_type='wealth'
                agent.net_need+=find_neighbours(agent)  
        
    return 0

def select_need():
    start=time.time()
    Calc_growth_need()
    end=time.time()
    function_time['calc_growth_need'].append(end-start)
    start=time.time()
    Calc_add_cells_need()
    end=time.time()
    function_time['Calc_add_cells_need'].append(end-start)
    start=time.time()
    Calc_maint_need()
    end=time.time()
    function_time['Calc_maint_need'].append(end-start)
    select_need_type()
    return 0
    
def pass_need(function):
    for agent in agents:
        function(agent,agent.net_need) 
    return 0     

def add_resource_to_cells(): #Pays cost for adding cell and adds resource to cell
    for agent in agents:
        if agent.need_type == 'maint':
            for cell in agent.cells:
                if cell.cost_paid == 'Y' and cell.resource<Optimum_resource and agent.keep>0:
                    cell_need = Optimum_resource-cell.resource
                    if cell_need<agent.keep:
                        cell.resource+=cell_need
                        agent.keep-=cell_need
                    else:
                        cell.resource+=agent.keep
                        cell_need=0
                        agent.keep=0    
    return 0

def add_cell():
    for agent in agents:
        if agent.need_type == 'add_cell':
            for cell in agent.cells:
                if cell.cost_paid == 'N' and agent.keep>cost_to_add_cell:
                    cell.cost_paid = 'Y'
                    agent.keep-=cost_to_add_cell
    return 0

def extend_check(agent,stoch,very_stoch,max_growth): #Checks if agents can add further agents
    next_chan_sel = []
    max_lib = []
    area=update_area()    
    check_area = deepcopy(area)
    after_zeros=0
    max_add=-9999
    add = -9999
    x,y = agent.pos
    for i in range(len(neumann)):
        check_area = deepcopy(area)
        a,b = neumann[i]
        Nx = x + a
        Ny = y + b  
        before_num,before_lib = check_zeros(area,Nx,Ny)  
        if area[Nx][Ny]==1:
            pass
        else:
            penalty=0
            if area[Nx][Ny]==2:
                penalty = check_if_last_RC((Nx,Ny))
            check_area[Nx][Ny]=1
            after_zeros,after_lib=check_zeros(check_area,Nx,Ny)
            after_zeros-=penalty
            if (after_zeros>before_num) and ((after_zeros-before_num)>=max_add) and ((after_zeros-before_num)>-9999):
                if stoch == 'Y':
                    max_add,next_chan_sel,Nx,Ny,after_zeros,before_num=stoch1(next_chan_sel,Nx,Ny,max_add,after_zeros,before_num)
                    if (i == (len(neumann)-1)) and (len(next_chan_sel)>1):
                        aa = randint(0,(len(next_chan_sel)-1))
                        add = next_chan_sel[aa]
                        xx,yy=add
                        check_area = deepcopy(area)
                        check_area[xx][yy]=1
                        after_zeros,after_lib=check_zeros(check_area,xx,yy)
                        after_zeros-=penalty
                        max_lib=after_lib
                
                    else:
                        add = next_chan_sel[0]
                        xx,yy=add
                        check_area = deepcopy(area)
                        check_area[xx][yy]=1
                        after_zeros,after_lib=check_zeros(check_area,xx,yy)
                        after_zeros-=penalty
                        max_lib=after_lib
                else:
                    add = (Nx,Ny)
                    max_add = after_zeros-before_num
                    xx,yy=add
                    check_area = deepcopy(area)
                    check_area[xx][yy]=1
                    after_zeros,after_lib=check_zeros(check_area,xx,yy)
                    after_zeros-=penalty
                    max_lib=after_lib 
    return len(max_lib),max_lib
    
def agent_extend_check(agent):
    stoch,very_stoch,max_growth = switch_growth_type()
    X,Y=agent.pos
    agent.ext_cells=[]
    checker,ext_lib1 = extend_check(agent,stoch,very_stoch,max_growth)
    checker = len(ext_lib1)
    agent.ext_agents=(ext_lib1)
    if checker>0:
        agent.extension='Y'
        agent.ext_amount=checker
    else:
        agent.extension='N'
        agent.ext_amount=checker      
    return 0
            
#########################Irrigation Network algoritm#########################################
def check_zeros(area,x,y):
    check_dic=[]
    check_ = []
    for check in range(len(Full_moore)):
      o,p=Full_moore[check]
      cx=x+o
      cy=y+p
      check_.append((cx,cy))
    check_set=set(check_) 
    for check in range(len(Full_moore)):
      o,p=Full_moore[check]
      cx=x+o
      cy=y+p
      if area[cx][cy]!=1:
          pass
      else:
          for check1 in range(len(Full_moore)):
              q,r=Full_moore[check1]
              cx1=cx+q
              cy1=cy+r
              if (cx1,cy1) not in check_set:
                  pass
              else:
                  if (area[cx1][cy1]==0):
                      checker = (cx1,cy1)
                      check_dic.append(checker)
    return len(check_dic),check_dic

def find_earliest_agent_can_grow(very_stoch,list_ag,next_cell): #Function which selects the next cell which can grow
    if next_cell == 'N':
        keep=0
        timestamp_check = len(agents)+1
        if very_stoch == 'Y':
            nn = 'N'
            while (nn == 'N'):
                ra = randint(0,(len(agents)-1))
                ag_check = agents[ra]
                nn = ag_check.extension
                keep = ag_check
        else:
            for ag in agents:
                if ag.extension == 'Y':
                    timestamp_check1=ag.creation
                    if timestamp_check1<timestamp_check:
                        timestamp_check=timestamp_check1   
                        keep = ag 
        return keep
    if next_cell == 'Y':
        keep=0
        timestamp_check = 9999999
        if very_stoch == 'Y':
            keep = list_ag[randint(0,len(list_ag)-1)]
        else:
            for ag in list_ag:
                if ag.extension == 'Y':
                    timestamp_check1=ag.creation
                    if timestamp_check1<timestamp_check:
                        timestamp_check=timestamp_check1   
                        keep = ag 
        return keep

def stoch1(next_chan_sel,Nx,Ny,max_add,after_zeros,before_num):
    if not next_chan_sel:
        next_chan_sel.append((Nx,Ny))
        max_add = after_zeros-before_num
    else:
        if (after_zeros-before_num)>max_add:
            next_chan_sel = []
            next_chan_sel.append((Nx,Ny))
            max_add = after_zeros-before_num
        if (after_zeros-before_num)==max_add:
            next_chan_sel.append((Nx,Ny))
            max_add = after_zeros-before_num
    return max_add,next_chan_sel,Nx,Ny,after_zeros,before_num

def check_if_last_RC(coords):
    for cell in cells:
        if cell.pos==coords:
            ag=cell.agent_
            if len(ag.cells)==1:
                return 3
            else:
                return 1
        
def next_channel(i_d,timestamp,stoch,very_stoch,next_cell,ag_to_expand,agent_num):
    next_chan_sel = []
    area=update_area()
    ag=find_earliest_agent_can_grow(very_stoch,ag_to_expand,next_cell)
    check_area = deepcopy(area)
    after_zeros=0
    max_add=-9999
    add = -9999
    x,y = ag.pos
    for i in range(len(neumann)):
        
        check_area = deepcopy(area)
        a,b = neumann[i]
        Nx = x + a
        Ny = y + b  
        before_num,before_lib = check_zeros(area,Nx,Ny)            
        if area[Nx][Ny]==1:
            pass
        else:
            penalty=0
            if area[Nx][Ny]==2:
                penalty=check_if_last_RC((Nx,Ny))
            check_area[Nx][Ny]=1
            after_zeros,after_lib=check_zeros(check_area,Nx,Ny)
            after_zeros-=penalty
            if (after_zeros>before_num) and ((after_zeros-before_num)>max_add) and ((after_zeros-before_num)>-9999):
                if stoch == 'Y':
                    max_add,next_chan_sel,Nx,Ny,after_zeros,before_num=stoch1(next_chan_sel,Nx,Ny,max_add,after_zeros,before_num)
                    if (i == (len(neumann)-1)) and (len(next_chan_sel)>1):
                        aa = randint(0,(len(next_chan_sel)-1))
                        add = next_chan_sel[aa]
                        xx,yy=add
                        check_area = deepcopy(area)
                        check_area[xx][yy]=1
                        after_zeros,after_lib=check_zeros(check_area,xx,yy)
                        max_lib=after_lib
                    else:
                        add = next_chan_sel[0]
                        xx,yy=add
                        check_area = deepcopy(area)
                        check_area[xx][yy]=1
                        after_zeros,after_lib=check_zeros(check_area,xx,yy)
                        max_lib=after_lib
                else:
                    add = (Nx,Ny)
                    max_add = after_zeros-before_num
                    xx,yy=add
                    check_area = deepcopy(area)
                    check_area[xx][yy]=1
                    after_zeros,after_lib=check_zeros(check_area,xx,yy)
                    max_lib=after_lib
    
    if max_add==-9999 and add == -9999:
        ag.extension = 'N' #This code removes agents which cannot extend from the list of agents which can expand
        ag_to_expand.remove(ag)
        return ag_to_expand,timestamp,agent_num
    else:
        Nx,Ny=add
        ag.growth_count+=1
        ag.keep-=cost_to_add_agent
        if agent_heterogeniety == 'Y':
            agents.append(agent([Nx,Ny],[x,y],ag,timestamp,Kin_Group_pref(),0,0, \
                                'Y',timestamp,Initial_resource_per_agent,0,0,0,[],[], \
                                [],0,0,{},[],0,0,0,0,0,0,0,[],0,0,float(np.random.normal(ag.ind_preference,SD,1)), \
                                0,neighbour_num,[],0,0,0,agent_num,ag.upstream_agents+1))
        elif agent_heterogeniety == 'N':
            agents.append(agent([Nx,Ny],[x,y],ag,timestamp,Kin_Group_pref(),0,0, \
                                'Y',timestamp,Initial_resource_per_agent,0,0,0,[],[], \
                                [],0,0,{},[],0,0,0,0,0,0,0,[],0,0,1.0, \
                                0,neighbour_num,[],0,0,0,agent_num,ag.upstream_agents+1))            
            
        agent_num+=1
        ag.downstream_agents.append(agents[-1])
        added_agent = agents[-1]
        ag_to_expand.remove(ag)
        for c in max_lib:
            x,y=c
            cells.append(cell((x,y),(Nx,Ny),added_agent,Initial_resource_per_cell,'N'))
            added_agent.cells.append(cells[-1])
        for c in cells:
            if c.pos==(Nx,Ny):
                agent1 = c.agent_
                if c in agent1.cells: 
                    agent1.cells.remove(c) 
                cells.remove(c) 
    timestamp+=1
    return ag_to_expand,timestamp,agent_num

def next_channel_max_growth(i_d,timestamp,stoch,very_stoch,next_cell,ag_to_expand):
    after_zeros=0
    max_add=-9999
    add = -9999
    add_select = []
    prev_ag = []
    for ag in ag_to_expand:
        next_chan_sel = []
        area=update_area()
        check_area = deepcopy(area)
        x,y = ag.pos
        for i in range(len(neumann)):
            check_area = deepcopy(area)
            a,b = neumann[i]
            Nx = x + a
            Ny = y + b  
            before_num,before_lib = check_zeros(area,Nx,Ny)            
            if area[Nx][Ny]==1:
                pass
            else:
                penalty=0
                if area[Nx][Ny]==2:
                    penalty = 1 
                check_area[Nx][Ny]=1
                after_zeros,after_lib=check_zeros(check_area,Nx,Ny)
                after_zeros-=penalty
                if (after_zeros>before_num) and ((after_zeros-before_num)>=max_add) and ((after_zeros-before_num)>-9999):
                    if stoch == 'Y':
                        max_add,next_chan_sel,Nx,Ny,after_zeros,before_num=stoch1(next_chan_sel,Nx,Ny,max_add,after_zeros,before_num)
                        if (i == (len(neumann)-1)) and (len(next_chan_sel)>1):
                            aa = randint(0,(len(next_chan_sel)-1))
                            add = next_chan_sel[aa]
                            add_select.append(add)
                            prev_ag.append(ag)
                            xx,yy=add
                            check_area = deepcopy(area)
                            check_area[xx][yy]=1
                            after_zeros,after_lib=check_zeros(check_area,xx,yy)
                            max_lib=after_lib
                    
                        else:
                            add = next_chan_sel[0]
                            add_select.append(add)
                            prev_ag.append(ag)
                            xx,yy=add
                            check_area = deepcopy(area)
                            check_area[xx][yy]=1
                            after_zeros,after_lib=check_zeros(check_area,xx,yy)
                            max_lib=after_lib
                    else:
                        add = (Nx,Ny)
                        add_select.append(add)
                        prev_ag.append(ag)
                        max_add = after_zeros-before_num
                        xx,yy=add
                        check_area = deepcopy(area)
                        check_area[xx][yy]=1
                        after_zeros,after_lib=check_zeros(check_area,xx,yy)
                        max_lib=after_lib
    if len(add_select)>0:
        aa = randint(0,(len(add_select)-1))
        add = add_select[aa]
        xx,yy=add
        prev_agg = prev_ag[aa]
        check_area = deepcopy(area)
        check_area[xx][yy]=1
        after_zeros,after_lib=check_zeros(check_area,xx,yy)  
        max_lib=after_lib
        Nx,Ny=add
        prev_agg.growth_count+=1
        prev_agg.keep-=cost_to_add_agent
        agents.append(agent([Nx,Ny],[x,y],prev_agg,timestamp,Kin_Group_pref(),0,0,'Y',timestamp, \
                            Initial_resource_per_agent,0,0,0,[],[],[],0,0,{},[],0,0,0,0,0,0,0,[],0, \
                            0,float(np.random.normal(prev_agg.ind_preference,SD,1)),0,neighbour_num,[],0,0,0,agent_num,prev_agg.upstream_agents+1)) 
        prev_agg.downstream_agents.append(agents[-1])
        added_agent = agents[-1]
        ag_to_expand.remove(prev_agg)
        for c in max_lib:
            x,y=c
            cells.append(cell((x,y),(Nx,Ny),added_agent,Initial_resource_per_cell,'N'))
            added_agent.cells.append(cells[-1])
        for c in cells:
            if c.pos==(Nx,Ny):
                agent1 = c.agent_
                if c in agent1.cells: 
                    agent1.cells.remove(c) 
                cells.remove(c) 
        
    elif max_add==-9999 and add == -9999:
        ag.extension = 'N' #This code removes agents which cannot extend from the list of agents which can expand
        ag_to_expand.remove(ag)
        return ag_to_expand,timestamp

    timestamp+=1
    return ag_to_expand,timestamp

#############################################################################################            

###############################Irrigation Area###############################################

def update_area():
    area = np.zeros((arrayx,arrayy),dtype=int)
    for agent in agents:
        x,y=agent.pos
        area[x][y]=1
    for cell in cells:
        x,y=cell.pos
        area[x][y]=2
    return area

           
#########################Agent algorithms####################################################            

def Kin_Group_pref():
    return float(randint(1,2))
            
def switch_growth_type():
    if growth_type == 0:
        stoch = 'N'
        very_stoch = 'N'
        max_growth = 'N'
    if growth_type == 1:
        stoch = 'Y'
        very_stoch = 'N'
        max_growth = 'N'   
    if growth_type == 2: 
        stoch = 'N'
        very_stoch = 'Y'
        max_growth = 'N' 
    if growth_type == 3: 
        stoch = 'N'
        very_stoch = 'N'
        max_growth = 'Y' 
    if growth_type == 4:
        s = randint(0,1)
        if s==0:
            stoch = 'Y'
        else:
            stoch = 'N'
        a = randint(0,1) 
        if a==0:
            very_stoch = 'Y'
        else:
            very_stoch = 'N'
        t = randint(0,1)   
        if t==0:
            max_growth = 'Y'
        else:
            max_growth = 'N'    
    return stoch,very_stoch,max_growth       
        
def assign_agents_flows(no_agents,timestamp,agent_num):
    stoch,very_stoch,max_growth = switch_growth_type()
    No_cells = []
    next_cell ='N'
    for i in range(no_agents):
        switch_growth_type()
        if i == 0: #If first cell, initial configuration is set up
            X,Y = initial_coordinates
            PrevX,PrevY = intial_previous_coordinates
            agents.append(agent([X,Y],[PrevX,PrevY],initial_prev_agent,timestamp,Kin_Group_pref(),0,0,'Y', \
                                timestamp,Initial_resource_per_agent,0,0,0,[],[],[],0,0,{},[],0,0,0,0,0,0,0,[], \
                                0,0,1.0,0,neighbour_num,[],0,0,0,agent_num,0))
            agent_num+=1
            area=update_area()
            after_zeros,after_lib=check_zeros(area,X,Y) 
            added_agent=agents[-1]
            for c in after_lib:
                x,y=c
                cells.append(cell((x,y),(X,Y),added_agent,0,'N'))
                added_agent.cells.append(cells[-1])
            timestamp+=1  
        
        elif max_growth == 'Y':
            timestamp=next_channel_max_growth(i,timestamp,stoch,very_stoch,next_cell,0)
        else:
            timestamp=next_channel(i,timestamp,stoch,very_stoch,next_cell,0)
        
        No_cells.append(len(cells))

    return No_cells,agent_num

def add_more_agents(timestamp,agent_num):
    stoch,very_stoch,max_growth = switch_growth_type()
    agents_to_expand = []
    i=0
    next_cell ='Y'
    for agent in agents:
        if agent.need_type=='add_agent':
            agent.growth_count=0
            if agent.extension == 'Y':
                if (agent.keep>=(cost_to_add_agent)):
                    agents_to_expand.append(agent)
    i = len(agents)
    if max_growth2 == 'N':
        while agents_to_expand:
            if growth_type == 3:
                agents_to_expand,timestamp=next_channel_max_growth(i,timestamp,stoch,very_stoch,next_cell,agents_to_expand)
            else:
                agents_to_expand,timestamp,agent_num=next_channel(i,timestamp,stoch,very_stoch,next_cell,agents_to_expand,agent_num)
    elif agents_to_expand:
        agents_to_expand,timestamp=next_channel(i,timestamp,stoch,very_stoch,next_cell,agents_to_expand)        
    return timestamp,agent_num


#########################Add Resource Functions#################################################
    
def add_res():
    downstream_extension()
    for agent in agents:
        if agent.pos == (initial_coordinates):
            agent.pass_+=Resource_added_per_timestep
        elif agent.extension=='Y' or agent.ext_downstream=='Y':
            agent.pass_+=Initial_resource_per_agent_per_timestep
    return 0
          
def use_resource(agents,cells):
    if cells:
        for cell in reversed(cells):
            if cell.resource>0:
                cell.resource-=1
    return 0

def delete_cells(agents,cells):
    for cell in reversed(cells):
        if cell.resource<=0:
            cell.cost_paid = 'N'
    return 0

def check_adj_to_agent(agent,area):
    return 0

def delete_agents():
    for c1 in cells:
        if c1.resource<=0:
            c1.cost_paid=='N'
    for agent in reversed(agents):
        if agent.pos==initial_coordinates:
            pass
        else:
#        area=update_area()
            if agent.cells:
                no_cells = len(agent.cells)
                no_cost_cells = 0
                for c1 in agent.cells:
                    if c1.cost_paid == 'N':
                        no_cost_cells+=1
                if no_cost_cells>=(no_cells/2):
    #            if no_cost_cells==no_cells and len(agent.downstream_agents)==0:         
                    agent.death_count+=1
                else: 
                    agent.death_count=0
            if not agent.cells:
                agent.death_count+=1
            if agent.death_count==10:
                if type(agent.prev_agent)!=int:
                    ag2 = agent.prev_agent
                    ag2.downstream_agents.remove(agent)
                    ag2.extension='Y'
                    cx,cy = agent.pos
                    Nx,Ny=ag2.pos
                    cells.append(cell((cx,cy),(Nx,Ny),ag2,0,'N'))
                    ag2.cells.append(cells[-1])
                for agg in agent.downstream_agents:
                    agg.prev_agent=0
                cells_=[]
                if not agent.cells:
                    agents.remove(agent)
                elif agent.cells:
                    for c in agent.cells:
                        cells_.append(c)
                    agents.remove(agent)
                    for c in reversed(cells_):
                        c_x,c_y = c.pos
                        c_moore = set()
                        for i in moore:
                            xx,yy=i
                            c_moore.add((c_x+xx,c_y+yy))
                        for agg in agents:
                            pos3,pos2=agg.pos
                            pos1=(pos3,pos2)
                            if pos1 in c_moore:
                                agg.cells.append(c)
                                c.agent_loc=agg.pos
                                c.agent_=agg
                                cells_.remove(c)
                                break
                            else:
                                pass
                    if cells_:
                        for c in cells_:
                            cells.remove(c)

    return 0

#########################Economic Analysis######################################################

def gini_ind_pref(agents):
    agent_max = 0
    agents_wealth = []
    agents_wealth_2 = []
    ind_pref=[]
    for agent in agents:
        ind_pref.append(agent.ind_preference)
        num_rcs=0
        for cc in agent.cells:
            if cc.resource>0:
                num_rcs+=1
        agents_wealth.append(agent.keep+num_rcs)
        if agent.keep>agent_max:
            agent_max=agent.keep
    pref_lib.append(ind_pref)
    agents_wealth_2 = deepcopy(agents_wealth)
    total = 0
    for i in range(len(agents_wealth)):
        x = agents_wealth[i]
        for j in range(len(agents_wealth_2)):
            y = agents_wealth_2[j]
            total+= abs(x-y)
    if sum(agents_wealth)==0:
        pass
    else:
        G = float(total)/(2*(len(agents_wealth))*sum(agents_wealth))
        ginilib.append(G)
    min_agent_wealth.append(min(agents_wealth))
    max_agent_wealth.append(max(agents_wealth))
    
    return 0
            
#######################Plot agents/flows########################################################

def Plot_area(area):
    number = np.amax(area)+10
    fig, ax = plt.subplots()
    plt.xlim(0,arrayx+0.5)
    plt.ylim(0,arrayy+0.5)
    colors = [(1.0,1.0,1.0)]
    colors.extend(mpl.cm.Greys(np.linspace(0, 1, number)))
    cmap = mpl.colors.ListedColormap(colors)     
    ax.set_xticks(np.arange(-.5, arrayx, 1), minor=True)
    ax.set_yticks(np.arange(-.5, arrayy, 1), minor=True)
    ax.grid(which='minor', color='grey', linestyle='-', linewidth=0.25)
    No_cells=0
    for cell in cells:
        if cell.cost_paid=='Y':
            No_cells+=1
    ax.set_xlabel("Distribution Cells: {0}. Receiver Cells: {1}".format(len(agents),No_cells))
    ax.imshow(area,interpolation='none',origin='lower',cmap=cmap,aspect='equal')

#    plt.scatter(coordinatesx,coordinatesy)
    plt.axis('on')
    fig.savefig('system1.png', format='png', dpi=1000)
#    plt.close() #switch on and off to print figures
    return 0   

def Plot_agents(agents,cells,ts,n_path):
    area = np.zeros((arrayx,arrayy),dtype=int)
    for agent in agents:
        x,y=agent.pos
        area[x][y]=20
    for cell in cells:
        if cell.cost_paid=='Y':
            x,y=cell.pos 
            area[x][y]=10
    number = np.amax(area)+10
    fig, ax = plt.subplots()
    plt.xlim(0,arrayx+0.5)
    plt.ylim(0,arrayy+0.5)
    colors = [(1.0,1.0,1.0)]
    colors.extend(mpl.cm.Greys(np.linspace(0, 1, number)))
    cmap = mpl.colors.ListedColormap(colors)     
    ax.set_xticks(np.arange(-.5, arrayx, 1), minor=True)
    ax.set_yticks(np.arange(-.5, arrayy, 1), minor=True)
    No_cells=0
    for cell in cells:
        if cell.cost_paid=='Y':
            No_cells+=1
    ax.grid(which='minor', color='grey', linestyle='-', linewidth=0.25)
    ax.set_xlabel("Distribution Cells: {0}. Receiver Cells: {1}".format(len(agents),No_cells))
    ax.imshow(area,interpolation='none',origin='lower',cmap=cmap,aspect='equal')

#    plt.scatter(coordinatesx,coordinatesy)
    plt.axis('on')
    if ts==1:
        name = "%i.png" % ts
        fig.savefig(os.path.join(n_path,name),format='png')
    if ts%25==0:
        name = "%i.png" % ts
        fig.savefig(os.path.join(n_path,name),format='png')

    plt.show()
    plt.close()
    return 0   

def x_y_plot(df1,No_sims):    
    fig, ax = plt.subplots()             
    plt.xlabel('Number of Distribution Cells')
    plt.ylabel('Number of Receiver Cells')
    colors = [(1.0,1.0,1.0)]
    colors.extend(mpl.cm.Greys(np.linspace(0, 1,len(df1.columns)-1)))
    c=1
    for i in range(No_sims):
        x = df1['No. Distrib Cells']
        y = df1[i]
        plt.plot(x,y,linewidth=0.25,color=colors[c]) 
        mpl.rcParams.update({'font.size': 10})
        c+=1
        
    plt.savefig('check2.png', dpi = 300)
    plt.show()   
    df1=df1.drop('No. Distrib Cells', 1)
    distrib_50=(df1.iloc[-1])
    plt.hist(distrib_50,edgecolor='black',facecolor='none',bins=np.arange(min(distrib_50), max(distrib_50) + 1, 1))
    plt.xlabel('Number of Receiver Cells for 50 Distribution Cells')
    plt.ylabel('Frequency')
    plt.savefig('check3.png', dpi = 300)
    plt.show()

def x_y_plot1(DC,RC,n_path):    
    fig, ax = plt.subplots()             
    plt.xlabel('Time Step')
    plt.ylabel('Number of Cells')
    timesteps = list(range(0,len(DC)))
    plt.plot(timesteps,DC,linewidth=0.5)
    plt.plot(timesteps,RC,linewidth=0.5)
    plt.savefig(os.path.join(n_path,'num_agents.png'), dpi = 300)
    plt.show()
    
def x_y_plot2(ginilib,n_path):    
    fig, ax = plt.subplots()             
    plt.xlabel('Time Step')
    plt.ylabel('Gini Coefficient')
    timesteps = list(range(0,len(ginilib)))
    plt.plot(timesteps,ginilib,linewidth=0.5)
    plt.savefig(os.path.join(n_path,'gini.png'), dpi = 300)
    plt.show()
    
def x_y_plot3(pref_lib,n_path):
    fig, ax = plt.subplots()             
    plt.xlabel('Time Step')
    plt.ylabel('Individual Preference')    
    timesteps = list(range(0,len(pref_lib)))
    y_sum=[]
    for i in (timesteps):
        lib = pref_lib[i]
        y_sum.append(math.fsum(lib)/(len(lib)))
    plt.plot(timesteps,y_sum,linewidth=0.5)
    plt.savefig(os.path.join(n_path,'ind_pref.png'), dpi = 300)
    plt.show()
    
def x_y_plot4(agent_data,n_path):
    fig, ax = plt.subplots()             
    plt.xlabel('Time Step')
    plt.ylabel('Agent Preference')    

    for dat in agent_data:
        x_=[]
        y_=[]
        for xy in dat:
            x,y=xy
            x_.append(x)
            y_.append(y)
        plt.plot(x_,y_,linewidth=0.5)
    plt.savefig(os.path.join(n_path,'pref_data.png'), dpi = 300)
    plt.show()    
    
def x_y_plot5(wealth_data,n_path):
    fig, ax = plt.subplots() 
    plt.yscale('log')            
    plt.xlabel('Time Step')
    plt.ylabel('Agent Resource')    

    for dat in wealth_data:
        x_=[]
        y_=[]
        for xy in dat:
            x,y=xy
            x_.append(x)
            y_.append(y)
        plt.plot(x_,y_,linewidth=0.5)
    plt.savefig(os.path.join(n_path,'wealth_data.png'), dpi = 300)
    plt.show()
    
def plot_sub_plots(area_lib,n_path,ts_graph,num_dc,num_rc):
    fig=plt.figure(figsize=(10,10))
    fig_no = 221
    ts = len(ts_graph)
    select_area=[]
    if ts>4:
        for i in range(0,ts,round(ts/3)):
            select_area.append(i)
        select_area.append(ts)
        ax1=plt.subplot(221)
        area = area_lib[select_area[0]]
        number = np.amax(area)+10
        plt.xlim(0,arrayx+0.5)
        plt.ylim(0,arrayy+0.5)
        colors = [(1.0,1.0,1.0)]
        colors.extend(mpl.cm.Greys(np.linspace(0, 1, number)))
        cmap = mpl.colors.ListedColormap(colors)     
        ax1.set_xticks(np.arange(-.5, arrayx, 1), minor=True)
        ax1.set_yticks(np.arange(-.5, arrayy, 1), minor=True)
    #        No_cells=0
    #        for cell in cells:
    #            if cell.cost_paid=='Y':
    #                No_cells+=1
        ax1.grid(which='minor', color='grey', linestyle='-', linewidth=0.25)
    #        ax.set_xlabel("Distribution Cells: {0}. Receiver Cells: {1}".format(len(agents),No_cells))
        ax1.imshow(area,interpolation='none',origin='lower',cmap=cmap,aspect='equal')
    #    divider = make_axes_locatable(ax1)
        ax1.set_xlabel("Distribution Cells: {0}. Receiver Cells: {1}".format(num_dc[select_area[0]],num_rc[select_area[0]]),fontsize=10)
        ax1.set_title("Time step: {0}".format(select_area[0]),fontsize=10)
        
        ax2=plt.subplot(222)
        area = area_lib[select_area[1]]
        number = np.amax(area)+10
        plt.xlim(0,arrayx+0.5)
        plt.ylim(0,arrayy+0.5)
        colors = [(1.0,1.0,1.0)]
        colors.extend(mpl.cm.Greys(np.linspace(0, 1, number)))
        cmap = mpl.colors.ListedColormap(colors)     
        ax2.set_xticks(np.arange(-.5, arrayx, 1), minor=True)
        ax2.set_yticks(np.arange(-.5, arrayy, 1), minor=True)
    #        No_cells=0
    #        for cell in cells:
    #            if cell.cost_paid=='Y':
    #                No_cells+=1
        ax2.grid(which='minor', color='grey', linestyle='-', linewidth=0.25)
    #        ax.set_xlabel("Distribution Cells: {0}. Receiver Cells: {1}".format(len(agents),No_cells))
        ax2.imshow(area,interpolation='none',origin='lower',cmap=cmap,aspect='equal')
        ax2.set_xlabel("Distribution Cells: {0}. Receiver Cells: {1}".format(num_dc[select_area[1]],num_rc[select_area[1]]),fontsize=10)
        ax2.set_title("Time step: {0}".format(select_area[1]),fontsize=10)
        
        ax3=plt.subplot(223)
        area = area_lib[select_area[2]]
        number = np.amax(area)+10
        plt.xlim(0,arrayx+0.5)
        plt.ylim(0,arrayy+0.5)
        colors = [(1.0,1.0,1.0)]
        colors.extend(mpl.cm.Greys(np.linspace(0, 1, number)))
        cmap = mpl.colors.ListedColormap(colors)     
        ax3.set_xticks(np.arange(-.5, arrayx, 1), minor=True)
        ax3.set_yticks(np.arange(-.5, arrayy, 1), minor=True)
    #        No_cells=0
    #        for cell in cells:
    #            if cell.cost_paid=='Y':
    #                No_cells+=1
        ax3.grid(which='minor', color='grey', linestyle='-', linewidth=0.25)
    #        ax.set_xlabel("Distribution Cells: {0}. Receiver Cells: {1}".format(len(agents),No_cells))
        ax3.imshow(area,interpolation='none',origin='lower',cmap=cmap,aspect='equal')
        ax3.set_xlabel("Distribution Cells: {0}. Receiver Cells: {1}".format(num_dc[select_area[2]],num_rc[select_area[2]]),fontsize=10)
        ax3.set_title("Time step: {0}".format(select_area[2]),fontsize=10)
    
        ax4=plt.subplot(224)
        area = area_lib[-1]
        number = np.amax(area)+10
        plt.xlim(0,arrayx+0.5)
        plt.ylim(0,arrayy+0.5)
        colors = [(1.0,1.0,1.0)]
        colors.extend(mpl.cm.Greys(np.linspace(0, 1, number)))
        cmap = mpl.colors.ListedColormap(colors)     
        ax4.set_xticks(np.arange(-.5, arrayx, 1), minor=True)
        ax4.set_yticks(np.arange(-.5, arrayy, 1), minor=True)
    #        No_cells=0
    #        for cell in cells:
    #            if cell.cost_paid=='Y':
    #                No_cells+=1
        ax4.grid(which='minor', color='grey', linestyle='-', linewidth=0.25)
    #        ax.set_xlabel("Distribution Cells: {0}. Receiver Cells: {1}".format(len(agents),No_cells))
        ax4.imshow(area,interpolation='none',origin='lower',cmap=cmap,aspect='equal')
        ax4.set_xlabel("Distribution Cells: {0}. Receiver Cells: {1}".format(num_dc[-1],num_rc[-1]),fontsize=10)
        ax4.set_title("Time step: {0}".format(select_area[-1]),fontsize=10)
    
        fig_no+=1
        fig.show()
        plt.savefig(os.path.join(n_path,'subplots.png'), dpi = 300)
    return 0

    def cells_per_agent():
        return 0

##########################################################################################
def cells_downstream(n_path):
    for agent in agents:
        down_agent=agent
        upstream = down_agent.prev_agent
        while type(upstream)!=int:
            upstream.total_downstream_agents+=1
            temp = upstream.prev_agent
            upstream=temp
    downstream_data = []
    for agent in agents:
        downstream_data.append(agent.total_downstream_agents)
    plt.yscale('log')
    plt.xscale('log')
    plt.hist(downstream_data,bins=np.arange(min(downstream_data), max(downstream_data) + 1, 1),edgecolor='black',facecolor='none')
    plt.xlabel('Number of Distribution Cells Downstream')
    plt.ylabel('Number of Distribution Cells')
    plt.savefig(os.path.join(n_path,'histo.png'), dpi = 300)
    plt.show()
    return 0

def wealth_distrib(n_path):
    wealth_data=[]
    for agent in agents:
        wealth_data.append(agent.keep)
    plt.yscale('log')
    plt.xscale('log')
    plt.hist(wealth_data,bins=np.arange(min(wealth_data), max(wealth_data) + 1, 1),edgecolor='black',facecolor='none')
    plt.xlabel('Wealth')
    plt.ylabel('Number of Distribution Cells')
    plt.savefig(os.path.join(n_path,'wealth_histo.png'), dpi = 300)
    plt.show()
    return 0

def animate(i,arraylib,im,tx):
    arr = arraylib[i]
    vmax     = np.max(arr)
    vmin     = np.min(arr)
    im.set_data(arr)
    im.set_clim(vmin, vmax+1)
    tx.set_text('Timestep {0}'.format(i))

def movie2(arraylib,area_max,n_path):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    div = make_axes_locatable(ax)
    cax = div.append_axes('right', '5%', '5%')
    cv0 = arraylib[0]
    number = np.amax(area)+10    
    colors = [(1.0,1.0,1.0)]
    colors.extend(mpl.cm.cool(np.linspace(0, 1, number)))
    cmap = mpl.colors.ListedColormap(colors)
    tx = ax.set_title('Frame 0')
    im = ax.imshow(cv0,interpolation='nearest',origin='lower',cmap=cmap) # Here make an AxesImage rather than contour
    fig.colorbar(im, cax=cax)

    ani = animation.FuncAnimation(fig, animate, frames=len(arraylib),fargs=(arraylib,im,tx))
    ani.save(os.path.join(n_path,'basic_animation.mp4'), fps=15, extra_args=['-vcodec', 'libx264'],dpi=100)
    plt.show()

def add_to_agent_lib(agent_lib,agent_max):
    area = np.zeros((arrayx,arrayy),dtype=int)
    
    for agent in agents:
        x,y=agent.pos
        area[x][y]=20
        agent_max=(20)
    for cell in cells:
        if cell.cost_paid=='Y':
            x,y=cell.pos 
            area[x][y]=10
    temp = deepcopy(area)
    agent_lib.append(temp)
    return agent_max

def sum_cell_resource():
    area = np.zeros((arrayx,arrayy),dtype=int)
    for cell in cells:
        x,y=cell.pos
        area[x][y]=cell.resource
    cell_resource = np.sum(area)
    return cell_resource            

def sum_res_agent_():
    Num_DCs.append(len(agents))
    No_cells=0
    total_res_in_sys = 0
    ts_graph.append(0)
    for cell in cells:
        if cell.resource>0:
            No_cells+=1
    Num_RCs.append(No_cells)
    tot_res=sum_cell_resource()
    total_resource_in_cells.append(tot_res)
    total_res_in_sys +=tot_res
    for agent in agents:
        total_res_in_sys+=agent.keep
        total_res_in_sys+=agent.pass_
    total_res_in_system.append(total_res_in_sys)
    return 0
    
def initialise(agents_flows,timestamp,agent_num): #Sets initial conditions for simulation  
    no_agents=agents_flows
    No_cells,agent_num=assign_agents_flows(no_agents,timestamp,agent_num)
    
    return No_cells,agent_num

def pref_lib_update():
    return 0

def save_data():
    new_path = '/Users/as18g13/Google Drive/02 Coding/00 Python/07 Irrigation models/Agent_Heterogeneity/Tests/resource_change/Growth_Type{}_resource_ts{}_neighbour_num{}_NS_{}_sim_num{}'.format(growth_type,Resource_added_per_timestep,neighbour_num,NS,sim_num)
    if not os.path.exists(new_path):
        os.makedirs(new_path)
    return new_path

def export_agent_data(n_path):
    with open(os.path.join(n_path,'agent_data.txt'), "w") as f:
        for ag in agent_data:
            for dd in ag:
                for ee in dd:
                    f.write(str(ee)+',')
            f.write('\n')
    with open(os.path.join(n_path,'wealth_data.txt'), "w") as f:
        for ag in wealth_data:
            for dd in ag:
                for ee in dd:
                    f.write(str(ee)+',')
            f.write('\n')
    with open(os.path.join(n_path,'upstream_agent_data.txt'), "w") as f:
        for ag in agent_upstream_data:
            for ee in ag:  
                f.write(str(ee)+',')
            f.write('\n')
    with open(os.path.join(n_path,'agent_network.txt'), "w") as f:
        for ag in agent_network:
            for dd in ag:
                for ee in dd:
                    f.write(str(ee)+',')
            f.write('\n')    
    return 0

def save_parameters(n_path):
    f = open(os.path.join(n_path,'parameters.txt'),'w')
    f.write('growth_type ={} \n'.format(growth_type))
    f.write('Initial_resource_per_agent ={} \n'.format(Initial_resource_per_agent))
    f.write('Initial_resource_per_cell ={} \n'.format(Initial_resource_per_cell))
    f.write('Initial_resource_per_agent_per_timestep ={} \n'.format(Initial_resource_per_agent_per_timestep))
    f.write('Resource_added_per_timestep={} \n'.format(Resource_added_per_timestep))
    f.write('cost_to_add_cell ={} \n'.format(cost_to_add_cell))
    f.write('cost_to_add_agent ={} \n'.format(cost_to_add_cell))
    f.write('Optimum_resource ={} \n'.format(Optimum_resource))
    f.write('initial_coordinates={} \n'.format(initial_coordinates))
    f.write('array_size={} \n'.format((arrayx,arrayy)))
    f.write('Initial_agent_number={} \n'.format(Initial_agent_number))
    f.write('time_steps={} \n'.format(time_steps))
    f.write('Neigbourhood size={} \n'.format(NS))   
    f.write('Simulation Number={} \n'.format(sim_num))
    f.write('Standard deviation={} \n'.format(SD))
    f.write('Neighbourhood Number={} \n'.format(neighbour_num))
    f.close()
    return 0

def time_functions():
    function_time['maint_select_need']=[]
    function_time['maint_pass_need']=[]
    function_time['maint_pass_down_resource']=[]
    function_time['maint_add_resource_to_cells']=[]
    function_time['add_cell_select_need']=[]
    function_time['add_cell_pass_need']=[]
    function_time['add_cell_pass_down_resource']=[]
    function_time['add_cell_add_cell']=[]    
    function_time['add_agent_select_need']=[]
    function_time['add_agent_pass_need']=[]
    function_time['add_agent_pass_down_resource']=[]
    function_time['add_agent_add_agent']=[]  
    function_time['calc_growth_need']=[]
    function_time['Calc_add_cells_need']=[]
    function_time['Calc_maint_need']=[]
    
    return 0

def collect_agent_data(ts,agent_num):
    if not agent_data:
        agent_data.append([])
    if not wealth_data:        
        wealth_data.append([])
    if not agent_upstream_data: 
        agent_upstream_data.append([])
    if not agent_network: 
        agent_network.append([])
    for agent in agents:
        if agent.pos==initial_coordinates:
            pass
        else:
            try:
                agent_data[agent.agent_ID].append((ts,float(agent.ind_preference)))
                wealth_data[agent.agent_ID].append((ts,float(agent.keep)))
                if not agent_upstream_data[agent.agent_ID]:
                    agent_upstream_data[agent.agent_ID].append(agent.upstream_agents)
                agn1=[]
                for agn in agent.neighbours:
                    agn1.append(agn.agent_ID)
                agent_network[agent.agent_ID].append((ts,agn1))                
                
            except:
                agent_data.append([])
                wealth_data.append([])
                agent_upstream_data.append([])
                agent_network.append([])
                if agent.agent_ID!=(len(agent_data)-1):
                    print('no')
                else:
                    agent_data[agent.agent_ID].append((ts,float(agent.ind_preference)))
                    wealth_data[agent.agent_ID].append((ts,float(agent.keep)))
                    if not agent_upstream_data[agent.agent_ID]:
                        agent_upstream_data[agent.agent_ID].append(agent.upstream_agents)
                    agn1=[]
                    for agn in agent.neighbours:
                        agn1.append(agn.agent_ID)
                    agent_network[agent.agent_ID].append((ts,agn1))  
    return 0

def add_network():
    print(len(agents))
    for agent in agents:
        print('here')
        print(agent.neighbours)
    """
    G,pos,n_col,edge=net1.initialise()
    ags = net1.find_edges(agents)
    coords= set([])
    for ag in ags:
        for PoS in range(len(ag.neighbours)):
            coords.add(ag.neighbours[PoS])
    for ag in ags:
        if ag.pos not in coords:
            pass
    G1,Alist1,Alist2,Alist3,Apos=net1.network(ags,G,pos)    
    
    temp1=deepcopy(G1)
    temp2=deepcopy(Alist1)
    temp3=deepcopy(Alist2)
    temp4=deepcopy(Alist3)
    temp5=deepcopy(Apos)
    Glib.append(temp1)
    list1.append(temp2)
    list2.append(temp3)
    list3.append(temp4)
    poslib.append(temp5)
    
    return Glib,list1,list2,list3,poslib
    """
    return 0

def main():
    from global_parameters2 import growth_type,Initial_resource_per_agent, \
    Initial_resource_per_cell, Resource_added_per_timestep,cost_to_add_cell,cost_to_add_agent, \
    Optimum_resource,initial_coordinates,arrayx,arrayy,intial_previous_coordinates,initial_prev_agent, \
    Initial_agent_number,time_steps,Initial_resource_per_agent_per_timestep,NS,sim_num,SD,neighbour_num
    df = pd.DataFrame()
    df['No. Distrib Cells'] = np.arange(1,Initial_agent_number+1)   
    n_path=save_data()
    save_parameters(n_path)
    time_functions()
    for i in range(Num_simulations):
        agent_max = 0
        timestamp = 1
        agent_num = 0        
        No_cells,agent_num=initialise(Initial_agent_number,timestamp,agent_num) #Initialise the model with the number of agents and flows
        agent_max=add_to_agent_lib(agent_lib,agent_max)
        sum_res_agent_()
        for ts in range(1,time_steps):
            print(ts)
            add_res()
            #maintenance
            select_need()
            from_pass_to_keep()
            clear_pass_down()
            pass_need(culumative_need)
            pass_down_resource()
            pass_down_res()
            add_resource_to_cells() 
            delete_cells(agents,cells)
            #add cells
            select_need()
            from_pass_to_keep()
            clear_pass_down()
            pass_need(culumative_need)
            pass_down_resource()
            pass_down_res()
            add_cell()
            #add agents
            select_need()
            from_pass_to_keep()
            clear_pass_down()
            pass_need(culumative_need)
            pass_down_resource()
            pass_down_res()
            timestamp,agent_num=add_more_agents(ts,agent_num)
            if max_growth2 == 'Y':
                prev_total_cells = 0
                prev_total_agents = 0
                total_cells = len(cells)
                total_agents = len(agents)
                count = 0
                while total_cells>prev_total_cells or total_agents>prev_total_agents: 
                    count+=1
                    prev_total_cells=len(cells)
                    prev_total_agents=len(agents)
                    select_need()
                    from_pass_to_keep()
                    clear_pass_down()
                    pass_need(culumative_need)
                    pass_down_resource()
                    add_resource_to_cells() 
                    #add cells
                    select_need()
                    from_pass_to_keep()
                    clear_pass_down()
                    pass_need(culumative_need)
                    pass_down_resource()
                    add_cell()
                    #add agents
                    select_need()
                    from_pass_to_keep()
                    clear_pass_down()
                    pass_need(culumative_need)
                    pass_down_resource()
                    timestamp=add_more_agents(ts)
                    total_cells = len(cells)
                    total_agents = len(agents)  
  
                    if count==50:
                        break

            use_resource(agents,cells)
            
            pref_lib_update()
#            for agent in agents:
#                if agent.pos==initial_coordinates:
#                    agent.pass_=0
#                if agent.keep<1:
#                    agent.keep=0
#                if agent.keep<0 or agent.pass_<0:
#                    print('negative',agent.pos,agent.keep,agent.pass_,agent.ind_maint,agent.ind_add_cell,agent.ind_add_agent,agent.need_type)
#                if agent.pass_>0 or agent.keep>50:
#                    print('positive',agent.pos,agent.keep,agent.neigh_ave,agent.net_need,agent.pass_,agent.ind_maint,agent.ind_add_cell,agent.ind_add_agent,agent.need_type,agent.extension,agent.ext_downstream ,agent.ind_preference,len(agent.neighbours))
#                if not agent.prev_agent:
#                    print(agent.pos,agent.keep,len(agent.cells))
#                    total_res=0
#                    for cell in agent.cells:
#                        total_res+=cell.resource
#                    print(total_res)
            gini_ind_pref(agents)
            delete_agents()
            agent_max=add_to_agent_lib(agent_lib,agent_max)
            collect_agent_data(ts,agent_num)
            Plot_agents(agents,cells,ts,n_path)
            sum_res_agent_()
        cells_downstream(n_path)
        wealth_distrib(n_path)
        movie2(agent_lib,agent_max,n_path)
        plot_sub_plots(agent_lib,n_path,ts_graph,Num_DCs,Num_RCs)
#        Plot_agents(agents,cells)
        df[i]=No_cells
        print(len(agents))
#        add_network()
        del agents[:]
        del cells[:]
    x_y_plot1(Num_DCs,Num_RCs,n_path)
    x_y_plot2(ginilib,n_path)
    x_y_plot3(pref_lib,n_path)
    x_y_plot4(agent_data,n_path)
    x_y_plot5(wealth_data,n_path)
    pref_lib_1=[]
    timesteps = list(range(0,len(pref_lib)))
    for i in (timesteps):
        lib = pref_lib[i]
        pref_lib_1.append(math.fsum(lib)/(len(lib)))
    with open(os.path.join(n_path,'global_data.txt'), "w") as output:
        writer = csv.writer(output)        
#        writer.writerow(['ts_graph,Num_DCs,Num_RCs,total_resource_in_cells,tot_res_in_system '])
        writer.writerows(zip(ts_graph,Num_DCs,Num_RCs,total_resource_in_cells,total_res_in_system,ginilib,pref_lib_1))
    export_agent_data(n_path)
    output.close()
    with open(os.path.join(n_path,'pass_data.txt'), "w") as output:
        writer = csv.writer(output)        
#        writer.writerow(['ts_graph,Num_DCs,Num_RCs,total_resource_in_cells,tot_res_in_system '])
        writer.writerows(zip(upstream_res,downstream_res))
    with open(os.path.join(n_path,'function_times.txt'), "w") as output1:
        writer = csv.writer(output1)
        writer.writerows(zip(function_time['maint_select_need'],function_time['maint_pass_need'],\
                             function_time['maint_pass_down_resource'],function_time['maint_add_resource_to_cells'],\
                             function_time['add_cell_select_need'],function_time['add_cell_pass_need'],\
                             function_time['add_cell_pass_down_resource'],function_time['add_cell_add_cell'],\
                             function_time['add_agent_select_need'],function_time['add_agent_pass_need'],\
                             function_time['add_agent_pass_down_resource'],function_time['add_agent_add_agent']))
   
    with open(os.path.join(n_path,'select_need_functions.txt'), "w") as output2:
        writer = csv.writer(output2) 
        writer.writerows(zip(function_time['calc_growth_need'],function_time['Calc_add_cells_need'],\
                             function_time['Calc_maint_need']))
    return agents
if __name__ == "__main__":
    area=main() 