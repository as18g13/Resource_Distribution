# -*- coding: utf-8 -*-
"""
Created on Thu Mar 22 11:27:58 2018

Irrigation algoritm which as of March 24th 2018, 

This simulation looks at different growth rates and the effect on the system

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
import os as os
import csv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as clr
from copy import deepcopy
import matplotlib.animation as animation
from mpl_toolkits.axes_grid1 import make_axes_locatable
import time
import matplotlib as mpl
import networkx as nx
from random import randint
import timeit
from functools import reduce
from global_parameters import *
import importlib

#######################Parameters##############################################
moore = [(-1, -1),(-1, 0),(-1, 1),(0, -1),(0, 1),(1, -1),(1,0),(1,1)] 
Full_moore = [(-1, -1),(-1, 0),(-1, 1),(0, -1),(0,0),(0, 1),(1, -1),(1,0),(1,1)] 
neumann = [(-1,0),(0,1),(1,0),(0,-1)]
initial_coordinates = 250,250
arrayx =500
arrayy=500
area = np.zeros((arrayx,arrayy),dtype=int)
intial_previous_coordinates = -9999,-9999
Initial_agent_number = 1
#Resource_added_per_timestep = 400
#Optimum_resource = 10
agents = []
cells = []
Initial_Resource = Initial_agent_number
Resource_used_per_timestep = 1
min_limit = 0
stoch = 'N'
very_stoch = 'N'
#cost_to_add_cell = 50
#Initial_resource_per_agent = 61
time_steps = 20
pass_parameter = 1.0 #Must be a float
growth = 10
r_parameter = 1
max_growth = 'Y'

#########Defining Agents in system and resource flow cells###################################

class agent(): #Class which assigns agents to the system
    def __init__(self,POS,PREV,ID,ext,timestamp,res,pass_down,pass_,need,add_cells,cells_down,growth_count,ext_amount):
        self.pos = POS #Position of agent in the system
        self.prev = PREV #Position of previous agent in system
        self.id = ID #ID assign to agent
        self.creation = timestamp
        self.extension = ext
        self.keep = res
        self.pass_ = pass_
        self.pass_down = pass_down
        self.need = need
        self.add_cells = add_cells
        self.cells_down = cells_down
        self.growth_count = growth_count
        self.ext_amount=ext_amount
        
class cell():
    def __init__(self,pos,source_agent,amount):
        self.pos = pos
        self.source_agent = source_agent
        self.amount = amount
        
            
#############################################################################################
            
#########################Irrigation Network algoritm#########################################

def check_zeros(area,x,y): #Checks for the number of zeros in the neighbourhood
    check_dic=set([])
    check_set = set([])
    for check in range(len(Full_moore)):
      o,p=Full_moore[check]
      cx=x+o
      cy=y+p
      check_set.add((cx,cy))
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
                      check_dic.add(checker)
    return len(check_dic)    

def find_earliest_agent_can_grow(): #Finds the earliest cell which can grow in 
    
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
        
def next_channel(i_d,timestamp):
    next_chan_sel = []
    area=update_area()
    ag=find_earliest_agent_can_grow()
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
        before_num = check_zeros(area,Nx,Ny)            
        if area[Nx][Ny]==1:
            pass
        else:
            check_area[Nx][Ny]=1
            after_zeros=check_zeros(check_area,Nx,Ny)
            if (after_zeros>before_num) and ((after_zeros-before_num)>=max_add) and ((after_zeros-before_num)>-9999):
                max_add,next_chan_sel,Nx,Ny,after_zeros,before_num=stoch1(next_chan_sel,Nx,Ny,max_add,after_zeros,before_num)
                if stoch == 'Y':
                    max_add,next_chan_sel,Nx,Ny,after_zeros,before_num=stoch1(next_chan_sel,Nx,Ny,max_add,after_zeros,before_num)
                    if (i == (len(neumann)-1)) and (len(next_chan_sel)>1):
                        aa = randint(0,(len(next_chan_sel)-1))
                        add = next_chan_sel[aa]
                    else:
                        add = next_chan_sel[0]
                else:                  
                    add = (Nx,Ny)
                    max_add = after_zeros-before_num
    if max_add==-9999 and add == -9999:
        ag.extension = 'N'
        next_channel(i_d,timestamp)
    else:
        Nx,Ny=add
        agents.append(agent((Nx,Ny),(x,y),i,'Y',timestamp,0,0,0,0,{},0,0))
    #timestamp+=1
    return timestamp,i_d

######################Agent Flow Algorithms##################################################

def use_resource(agents):
    for agent in agents:
        if agent.keep>min_limit:
            agent.keep-=Resource_used_per_timestep
            if agent.pass_>0:
                agent.keep+=1
                agent.pass_-=1
    return 0

def find_earliest_ag_can_grow(list_ag):
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

def extend_check(agent):
    next_chan_sel = []
    area=update_area()
    check_area = deepcopy(area)
    after_zeros=0
    max_add=-9999   
    x,y = agent.pos

    for i in range(len(neumann)):
        check_area = deepcopy(area)
        a,b = neumann[i]
        Nx = x + a
        Ny = y + b  
        before_num = check_zeros(area,Nx,Ny)            
        if area[Nx][Ny]==1:
            pass
        else:
            check_area[Nx][Ny]=1
            after_zeros=check_zeros(check_area,Nx,Ny)
            if (after_zeros>before_num) and ((after_zeros-before_num)>=max_add) and ((after_zeros-before_num)>-9999):
                max_add,next_chan_sel,Nx,Ny,after_zeros,before_num=stoch1(next_chan_sel,Nx,Ny,max_add,after_zeros,before_num)
    return len(next_chan_sel)

def agent_extend_check():
    for agent in agents:
        checker = extend_check(agent)
        if checker>0:
            agent.extension='Y'
            agent.ext_amount=checker
        else:
            agent.extension='N'
            agent.ext_amount=checker
    return 0

def next_channel1(timestamp,ag_to_expand):
    next_chan_sel = []
    area=update_area()
    ag=find_earliest_ag_can_grow(ag_to_expand)
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
        before_num = check_zeros(area,Nx,Ny)            
        if area[Nx][Ny]==1:
            pass
        else:
            check_area[Nx][Ny]=1
            after_zeros=check_zeros(check_area,Nx,Ny)
            if (after_zeros>before_num) and ((after_zeros-before_num)>=max_add) and ((after_zeros-before_num)>-9999):
                max_add,next_chan_sel,Nx,Ny,after_zeros,before_num=stoch1(next_chan_sel,Nx,Ny,max_add,after_zeros,before_num)
                if stoch == 'Y':
                    max_add,next_chan_sel,Nx,Ny,after_zeros,before_num=stoch1(next_chan_sel,Nx,Ny,max_add,after_zeros,before_num)
                    if (i == (len(neumann)-1)) and (len(next_chan_sel)>1):
                        aa = randint(0,(len(next_chan_sel)-1))
                        add = next_chan_sel[aa]
                    else:
                        add = next_chan_sel[0]
                else:                  
                    add = (Nx,Ny)
                    max_add = after_zeros-before_num
    if max_add==-9999 and add == -9999:
        ag.extension = 'N'
        ii=0
        for aga in ag_to_expand:
            if aga.pos==(x,y):
                del ag_to_expand[ii]
            ii+=1
        return ag_to_expand,timestamp
    else:
       # timestamp = ag.creation+1 switch with timestamp+=1 on line 290 to change creation consecutively to one up from previous
        Nx,Ny=add
        ag.growth_count+=1
        ag.keep-=cost_to_add_cell
        agents.append(agent((Nx,Ny),(x,y),i,'Y',timestamp,Initial_resource_per_agent,0,0,0,{},0,ag.growth_count,0))
        ag.ext_amount=len(next_chan_sel)-1
        ext_=extend_check(agents[-1])
        agent2=agents[-1]
        agent2.ext_amount=ext_
        if agent2.ext_amount>0:
            agent2.extension='Y'
        else:
            agent2.extension='N'
        
        ii=0
        for aga in ag_to_expand:
            if (aga.pos==(x,y)):
                if max_growth=='Y':
                    if (aga.keep<(cost_to_add_cell+Optimum_resource)): 
                    #"""or (aga.growth_count>=growth)"""
                        del ag_to_expand[ii]
                else:
                    del ag_to_expand[ii]
            ii+=1
        
        return ag_to_expand,timestamp
    
def add_more_agents(agents,timestamp,max_agents):
    agents_to_expand = []
    for agent in agents:
        agent.growth_count=0
        if agent.extension == 'Y':
            if (agent.keep>=(cost_to_add_cell+Optimum_resource)):
                agents_to_expand.append(agent)
    while agents_to_expand: #and (len(agents)<max_agents): 
        agents_to_expand,timestamp = next_channel1(timestamp,agents_to_expand)
    return timestamp

def check(x,y,px,py,agent_need,ext,next_cell,prev_cell):
    if (x,y)== intial_previous_coordinates: #not in system
        pass
    else:
        for agent in agents:
            if agent.pos == (x,y):
                agent.cells_down+=1
                if (px,py) in agent.pass_down.keys():
                    amount,extend = agent.pass_down[(px,py)]
                    if (ext == 'Y') and (extend =='N'):
                        extend = 'Y'
                    amount+=agent_need
                    agent.pass_down[(px,py)]=(amount,extend)
                    px,py = agent.pos
                    prev_cell.append((px,py))
                    x,y = agent.prev
                    next_cell.append((x,y))
                    return next_cell,prev_cell
                else:
                    agent.pass_down[(px,py)] = (agent_need,ext)
                    px,py = agent.pos
                    prev_cell.append((px,py))
                    x,y = agent.prev
                    next_cell.append((x,y))
                    return next_cell,prev_cell
                
    return next_cell,prev_cell 

def Calc_agent_need():
    for ag in agents:
        ag.pass_down = {} 
        ag.cells_down = 0
    for ag1 in agents:
        px,py= ag1.pos #position of agent
        if ag1.extension == 'Y': #if agent can expand then more resources are allocated
            opt_res = Optimum_resource + (cost_to_add_cell*ag1.ext_amount) + Resource_used_per_timestep
        else:
            opt_res = Optimum_resource +Resource_used_per_timestep
        if ag1.keep<opt_res: #if the amount kept by the agent is less than the optimum resource then the agent need is updated.
                agent_need = opt_res-ag1.keep
                ag1.need= agent_need
        else:
            agent_need = 0
            ag1.need= 0
        mx,my=ag1.prev
        next_cell = []
        prev_cell = []            
        next_cell,prev_cell=check(mx,my,px,py,agent_need,ag1.extension,next_cell,prev_cell) #passes agent need up the system
        while next_cell:
            mx,my=next_cell[0]
            del next_cell[0]
            px,py = prev_cell[0]
            del prev_cell[0]
            next_cell,prev_cell = check(mx,my,px,py,agent_need,ag1.extension,next_cell,prev_cell) 
        
    return 0

def pass_to_keep(agent):
    if agent.pass_<=agent.need:
        agent.keep+=agent.pass_
        agent.pass_=0
        agent.need-=agent.pass_
    if agent.pass_>agent.need:
        agent.keep+=agent.need
        agent.pass_-=agent.need
        agent.need=0
    mx,my=agent.prev
    px,py=agent.pos
    next_cell = []
    prev_cell = []            
    next_cell,prev_cell=check(mx,my,px,py,agent.need,agent.extension,next_cell,prev_cell) #passes agent need up the system
    while next_cell:
        mx,my=next_cell[0]
        del next_cell[0]
        px,py = prev_cell[0]
        del prev_cell[0]
        next_cell,prev_cell = check(mx,my,px,py,agent.need,agent.extension,next_cell,prev_cell) 
    return agents
                
def pass_res_down(agent,next_cell):
    if agent.need>0:
        need = agent.need
        if agent.pass_<=need:
            agent.keep+=agent.pass_
            agent.pass_=0
        else:
            agent.keep+=need
            agent.pass_-=round(need,1)

    if agent.pass_down and agent.pass_>0:

        res = 0 
        extend = 0
        for k,v in agent.pass_down.items():
            res+=v[0]
            res= float(res)
        if agent.pass_>=res:
            res_check=round(agent.pass_-res,1)
            res1 = round(res_check*pass_parameter,1)
            
            for k,v in agent.pass_down.items():
                if v[1]=='Y':
                    extend+=1
            if extend>0:
                portion=round(float(res1)/extend,1)
            if extend==0:
                portion=0   
            sum_ = 0
            for k,v in agent.pass_down.items():
                for ag1 in agents:
                    if ag1.pos == k:
                        ag1.pass_+=round(v[0],1)
                        agent.pass_-=round(v[0],1)
                        sum_+=round(v[0],1)
                        if v[1] == 'Y':
                            ag1.pass_+=round(portion,1)
                            agent.pass_-=round(portion,1)
                            if agent.pass_<0.0:
                                agent.pass_=0
                        x,y = ag1.pos
                        next_cell.append((x,y))

            return next_cell

        else:
            #res2 = round(float(agent.pass_)*pass_parameter,1)
            res1=round(float(agent.pass_)/len(agent.pass_down),1)

        
            to_del = []
            for k,v in agent.pass_down.items():
                for ag1 in agents:
                    if ag1.pos == k:
                        if v[1]=='N':
                            if res1>v[0]:
                                ag1.pass_+=v[0]
                                res1-=v[0]
                                agent.pass_-=v[0]
                                to_del.append(k)
                                if agent.pass_<0.0:
                                    agent.pass_=0
                                x,y = ag1.pos
                                next_cell.append((x,y))
                            else:
                                ag1.pass_+=res1
                                agent.pass_-=res1

                                to_del.append(k)
                                if agent.pass_<0.0:
                                    agent.pass_=0
                                x,y = ag1.pos
                                next_cell.append((x,y))
                            
            if to_del:
                for i in range(len(to_del)):
                    ff = to_del[i]
                    del agent.pass_down[ff]
            if agent.pass_down:
                res1=round(float(agent.pass_)/len(agent.pass_down),1) 
                for k,v in agent.pass_down.items():
                    for ag1 in agents:
                        if ag1.pos == k:
                            agent.pass_ -= round(res1,1)
                            ag1.pass_+=round(res1,1)
                            if agent.pass_<0.0:
                                agent.pass_=0
                            x,y = ag1.pos
                            next_cell.append((x,y))
            return next_cell
    else:
        for k,v in agent.pass_down.items():
            next_cell.append(k)
    return next_cell

def start_pass_res_down(ts):
    for agent in agents:
        if agent.pos == initial_coordinates:
            x,y=agent.pos
            agent.pass_+=Resource_added_per_timestep
            next_cell = []
            next_cell = pass_res_down(agent,next_cell)
    return next_cell

def start_pass_res_down_in_model(ts):
    for agent in agents:
        if agent.pos == initial_coordinates:
            x,y=agent.pos
            next_cell = []
            next_cell = pass_res_down(agent,next_cell)
    return next_cell


def add_more_agents_in_loop(agents,timestamp,max_agents):
    agents_to_expand = []
    for agent in agents:

        if agent.extension == 'Y':
            if (agent.keep>=(cost_to_add_cell+Optimum_resource)):
#                if agent.growth_count<growth:
                    agents_to_expand.append(agent)
    while agents_to_expand: #and (len(agents)<max_agents): 
        agents_to_expand,timestamp = next_channel1(timestamp,agents_to_expand)
    return timestamp

def grow(ts,max_agent_add,agent_lib,area_max):
    count=1
    while count > 0:
        agent_extend_check()
        #plot_agents11(agents,ts)        
        before_pass=0
        for agg in agents:
            before_pass += agg.pass_
        next_cell = start_pass_res_down_in_model(ts)
        while next_cell:
            x,y=next_cell[0]
            del next_cell[0]
            for agent in agents:
                agent.pass_=round(agent.pass_,0)
                if agent.pos == (x,y):
                    if agent.pass_<=0:
                        for k,v in agent.pass_down.items():
                            next_cell.append(k) 
                    else:
                        pass_to_keep(agent)
                        next_cell = pass_res_down(agent,next_cell)  
        after_pass=0
        for agg1 in agents:
            after_pass+=agg1.pass_
        count = before_pass-after_pass
        ts=add_more_agents_in_loop(agents,ts,max_agent_add)
        agent_extend_check()
        Calc_agent_need()
        agent_lib,area_max=add_to_agent_lib(agent_lib,area_max) 
    return agent_lib,area_max

def grow1(ts,max_agent_add):
    next_cell = start_pass_res_down_in_model(ts)
    while next_cell:
        x,y=next_cell[0]
        del next_cell[0]
        for agent in agents:
            agent.pass_=round(agent.pass_,0)
            if agent.pos == (x,y):
                if agent.pass_<=0:
                    for k,v in agent.pass_down.items():
                        next_cell.append(k) 
                else:
                    pass_to_keep(agent)
                    next_cell = pass_res_down(agent,next_cell)
    ts=add_more_agents_in_loop(agents,ts,max_agent_add)
    agent_extend_check()
    Calc_agent_need()
    return 0

def delete_dead_agents(agents,ts):
    to_del=set()
    to_del_cells=set()
    for i in range(len(agents)):
        agent = agents[i]
        if agent.keep<=0:
            to_del.add(i)
            cx,cy=agent.pos
            x,y= agent.prev
            for ag in agents:
                if ag.pos == (x,y):
                    ag.extension ='Y'
            for i in range(len(cells)):
                cell = cells[i]
                if cell.source_agent==(cx,cy):
                    to_del_cells.add(i)
    for i in range(len(agents),0,-1):
        if i in to_del:
            del agents[i]
    for i in range(len(cells),0,-1):
        if i in to_del_cells:
            del cells[i]
    return 0

def check_cell_zeros(area,x,y):
    check_dic=[]
    for check in range(len(Full_moore)):
      o,p=Full_moore[check]
      cx=x+o
      cy=y+p

      if area[cx][cy]==0:
          check_dic.append((cx,cy))

    return check_dic  

def add_cells():
    for agent in agents:
        cell_area=update_cell_area()
        x,y=agent.pos
        check_cell_dic=check_cell_zeros(cell_area,x,y)
        if check_cell_dic:
            for i in range(len(check_cell_dic)):
                cx,cy=check_cell_dic[i]
                cells.append(cell((cx,cy),(x,y,),10))
                
def add_res():
    for agent in agents:
        if agent.pos == (initial_coordinates):
            agent.pass_+=Resource_added_per_timestep
        
#############################################################################################            

###############################Irrigation Area###############################################

def update_area():
    area = np.zeros((arrayx,arrayy),dtype=int)
    for agent in agents:
        x,y=agent.pos
        area[x][y]=1
    return area

def update_cell_area():
    cell_area = np.zeros((arrayx,arrayy),dtype=int)
    for cell in cells:
        x,y=cell.pos
        cell_area[x][y]=1
    return cell_area
           
#########################Agent algorithms####################################################            
            
def Kin_Group_pref():
    return float(randint(1,2))

def assign_agents_flows(no_agents,timestamp):
    for i in range(no_agents):
        if i == 0: #If first cell, initial configuration is set up
            X,Y = initial_coordinates
            PrevX,PrevY = intial_previous_coordinates
            agents.append(agent((X,Y),(PrevX,PrevY),i,'Y',timestamp,Initial_resource_per_agent,0,0,0,{},0,0,0))
            ag=agents[0]
            ag.ext_amount=extend_check(ag)
            i_d=i
        else:
            timestamp,i_d=next_channel(i,timestamp)
            for agg in agents:
                pass
             #   print agg.creation, agg.extension, i
           # area = next_channel(agents,flows,area,i)
    return timestamp,i_d

######################Check agents in sim##################################################

#Checks value of agents in simulation
def check_agents_flows():
    for i in range(len(agents)):
        Check_agent =  agents[i]
#        Check_flow = flows[i]
        
        print (Check_agent.pos,Check_agent.prev, \
        Check_agent.id,Check_agent.ind_kin_group, \
        Check_agent.neighbours, Check_agent.neighbours, \
        Check_agent.NH, Check_agent.creation, Check_agent.extension)
        
#        print Check_flow.pos, Check_flow.amount, \
#        Check_flow.prev, Check_flow.id, Check_flow.creation, Check_flow.extension
        
    return 0
    
#######################Plot agents/flows########################################################

def plot_agents11(agents,timestep):
    area = np.zeros((arrayx,arrayy),dtype=int)
    for agent in agents:
        x,y=agent.pos
        area[x][y]=agent.keep
    number = np.amax(area)
    fig, ax = plt.subplots()
    plt.xlim(0,arrayx)
    plt.ylim(0,arrayy)
    plt.xlabel('Agent Cells: Agent Resource. Timestep:'+str(timestep))
    colors = [(1.0,1.0,1.0)]
    colors.extend(mpl.cm.jet(np.linspace(0, 1, number)))
    cmap = mpl.colors.ListedColormap(colors) 
#    cmap = clr.ListedColormap(['white','black','red','blue','green'])
#    bounds = [-0.5,0.5,1.5,2.5,3.5,4.5]
#    norm = clr.BoundaryNorm(bounds,cmap.N)
#    ax.imshow(area,interpolation='nearest',origin='lower',cmap=cmap,norm=norm)
#    ax.grid(color='r', linestyle='-', linewidth=2)
    im=ax.imshow(area,interpolation='nearest',origin='lower',cmap=cmap)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)

    plt.colorbar(im, cax=cax)
#    plt.scatter(coordinatesx,coordinatesy)
    plt.axis('on')
    plt.show()

    return 0   
  
def plot_agents(agents,timestep,n_path):
    area = np.zeros((arrayx,arrayy),dtype=int)
    for agent in agents:
        x,y=agent.pos
        area[x][y]=agent.keep
    number = np.amax(area)
    fig, ax = plt.subplots()
    plt.xlim(0,arrayx)
    plt.ylim(0,arrayy)
    plt.xlabel('Agent Cells: Agent Resource. Timestep:'+str(timestep))
    colors = [(1.0,1.0,1.0)]
    colors.extend(mpl.cm.jet(np.linspace(0, 1, number)))
    cmap = mpl.colors.ListedColormap(colors) 
#    cmap = clr.ListedColormap(['white','black','red','blue','green'])
#    bounds = [-0.5,0.5,1.5,2.5,3.5,4.5]
#    norm = clr.BoundaryNorm(bounds,cmap.N)
#    ax.imshow(area,interpolation='nearest',origin='lower',cmap=cmap,norm=norm)
#    ax.grid(color='r', linestyle='-', linewidth=2)
    im=ax.imshow(area,interpolation='nearest',origin='lower',cmap=cmap)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)

    plt.colorbar(im, cax=cax)
#    plt.scatter(coordinatesx,coordinatesy)
    plt.axis('on')
    fig.savefig(os.path.join(n_path,'system2.jpg'), format='jpg', dpi=1000)
    
    return 0 

def plot_cells(cells,timestep):
    area = np.zeros((arrayx,arrayy),dtype=int)
    for cell in cells:
        x,y=cell.pos
        area[x][y]=cell.amount
    number = np.amax(area)
    fig, ax = plt.subplots()
    plt.xlim(0,arrayx)
    plt.ylim(0,arrayy)
    plt.xlabel('Cells: Agent Resource. Timestep:'+str(timestep))
    colors = [(1.0,1.0,1.0)]
    colors.extend(mpl.cm.jet(np.linspace(0, 1, number)))
    cmap = mpl.colors.ListedColormap(colors) 
#    cmap = clr.ListedColormap(['white','black','red','blue','green'])
#    bounds = [-0.5,0.5,1.5,2.5,3.5,4.5]
#    norm = clr.BoundaryNorm(bounds,cmap.N)
#    ax.imshow(area,interpolation='nearest',origin='lower',cmap=cmap,norm=norm)
#    ax.grid(color='r', linestyle='-', linewidth=2)
    im=ax.imshow(area,interpolation='nearest',origin='lower',cmap=cmap)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)

    plt.colorbar(im, cax=cax)
#    plt.scatter(coordinatesx,coordinatesy)
    plt.axis('on')
    fig.savefig('system2.jpg', format='jpg', dpi=1000)
    
    return 0 

def plot_agents1(agents):

    for agent in agents:
        x,y=agent.pos
        area[x][y]=1
#    number = np.amax(area)
    fig, ax = plt.subplots()
    plt.xlim(0,arrayx)
    plt.ylim(0,arrayy)
    plt.xlabel('Agent Cells: Timestep Created On Plot')
#    colors = [(1.0,1.0,1.0)]
#    colors.extend(mpl.cm.jet(np.linspace(0, 1, number)))
#    cmap = mpl.colors.ListedColormap(colors) 
    cmap = clr.ListedColormap(['grey','black','red','blue','green'])
    bounds = [-0.5,0.5,1.5,2.5,3.5,4.5]
    norm = clr.BoundaryNorm(bounds,cmap.N)
    ax.imshow(area,interpolation='nearest',origin='lower',cmap=cmap,norm=norm)
#    ax.grid(color='r', linestyle='-', linewidth=2)
#    ax.imshow(area,interpolation='nearest',origin='lower',cmap=cmap)
#    plt.scatter(coordinatesx,coordinatesy)
    plt.axis('on')
    fig.savefig('system2.jpg', format='jpg', dpi=1000)
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
    colors = [(1.0,1.0,1.0)]
    colors.extend(mpl.cm.jet(np.linspace(0, 1, area_max-1)))
    cmap = mpl.colors.ListedColormap(colors)
    tx = ax.set_title('Frame 0')
    im = ax.imshow(cv0,interpolation='nearest',origin='lower',cmap=cmap) # Here make an AxesImage rather than contour
    fig.colorbar(im, cax=cax)

    ani = animation.FuncAnimation(fig, animate, frames=len(arraylib),fargs=(arraylib,im,tx))
    ani.save(os.path.join(n_path,'basic_animation.mp4'), fps=30, extra_args=['-vcodec', 'libx264'],dpi=100)
    plt.show()

def add_to_agent_lib(agent_lib,area_max):
    area = np.zeros((arrayx,arrayy),dtype=int)
    
    for agent in agents:
        x,y=agent.pos
        area[x][y]=agent.keep
        current = agent.keep
        if current>area_max:
            area_max=current
    temp = deepcopy(area)
    agent_lib.append(temp)
    return agent_lib,area_max

##########################################################################################
            
def initialise(agents_,timestamp): #Sets initial conditions for simulation  
    no_agents=agents_
    timestamp,i_d=assign_agents_flows(no_agents,timestamp)
    
    return timestamp ,i_d

def save_data():
    new_path = '/Users/as18g13/Google Drive/02 Coding/00 Python/07 Irrigation models/Section 8 model/Tot_{}_cost_{}_optres_{}_max_grow_{}_'\
    'init_res_{}'.format(Resource_added_per_timestep,cost_to_add_cell,Optimum_resource,max_growth,Initial_resource_per_agent)
    if not os.path.exists(new_path):
        os.makedirs(new_path)
    return new_path

def main():
    from global_parameters import Resource_added_per_timestep,cost_to_add_cell,Optimum_resource,Initial_resource_per_agent
    print (Resource_added_per_timestep,cost_to_add_cell,Optimum_resource,max_growth,Initial_resource_per_agent)
    n_path=save_data()
    
    num_ag = [] #lists to gather data from the simulation
    max_ag = 0
    min_ag = 999999
    agent_lib = []
    area_max = 0
    ag_graph = []
    TS_graph = []
    Total_Res_lib = []
    Total_res = []
    Total_res1 = []
    Total_agents = []
    rate_change = []
    ts_rate_change = []
    sum_keep_lib=[]
    timestamp = 0
    timestamp,i_d=initialise(Initial_agent_number,timestamp) #Initialise the model with the number of agents and flows
    ag_graph.append(len(agents))
    TS_graph.append(0) 
    sum_keep=0
    Tot_Res = 0
    for i in range(len(agents)):
        agent=agents[i]
        Tot_Res += agent.keep
        Tot_Res += agent.pass_
        sum_keep += agent.keep
    Res_div_ag = round(float(Tot_Res)/(len(agents)),1)
    Total_res.append(Res_div_ag)
    Total_Res_lib.append(Tot_Res)
    for ts in range(1,time_steps):
        print (ts)
        add_res()
        max_agent_add = int(round(len(agents)*growth,0))
        agent_lib,area_max=add_to_agent_lib(agent_lib,area_max) 
        use_resource(agents) 
        timestamp=add_more_agents(agents,ts,max_agent_add)
        agent_extend_check()
        Calc_agent_need()
        if max_growth == 'Y':
            agent_lib,area_max=grow(ts,max_agent_add,agent_lib,area_max)
        else:
            grow1(ts,max_agent_add)
        delete_dead_agents(agents,ts)
        Tot_Res = 0
        sum_keep = 0
        for i in range(len(agents)):
            agent=agents[i]
            Tot_Res += agent.keep
            Tot_Res += agent.pass_
            sum_keep += agent.keep
        Res_div_ag = round(float(Tot_Res)/(len(agents)),1)
        Total_res.append(Res_div_ag)
        Total_Res_lib.append(Tot_Res)
        print (Tot_Res)
        
        Total_agents.append((len(agents)))
        ag_graph.append(len(agents))
        rate_change.append((float(ag_graph[ts])/ag_graph[ts-1]))
        ts_rate_change.append(ts)
        TS_graph.append(ts)
#        Calc_agent_need()
#        plot_flows(flows)
        """
        if ts == 50 or ts == 100:
            plot_agents(agents,ts) 
#            
        if ts % 250==0:
            plot_agents(agents,ts)
        if ts % 499==0:
            plot_agents(agents,ts)
        """
        if ts>300:
            if sum_keep<min_ag:
                min_ag=sum_keep
            if sum_keep>max_ag:
                max_ag=sum_keep
            num_ag.append(float(len(agents)))
            Total_res1.append(float(Tot_Res))
            sum_keep_lib.append(sum_keep)

        
    plot_agents(agents,ts,n_path)
    plot_cells(cells,ts)
    fig, ax = plt.subplots()
    plt.Line2D(TS_graph,ag_graph)
    xmax=(max(TS_graph)+10)
    ymax = (max(ag_graph)+50)
    plt.xlim(0,xmax)
    plt.ylim(0,ymax)
    plt.xlabel('Time Step')
    plt.ylabel('Number of Agent Cells')
    plt.plot(TS_graph,ag_graph,color='black')
    fig.savefig((os.path.join(n_path,'ag_per_ts.png')))
    with open(os.path.join(n_path,'agents.txt'), "w") as output:
        writer = csv.writer(output)
        writer.writerows(zip(TS_graph,ag_graph,Total_Res_lib,rate_change))
    
    fig, ax = plt.subplots()
    plt.xlim(0,(max(ts_rate_change)+10) )
    plt.xlabel('Time step')
    plt.ylabel('Rate of change')
    plt.plot(ts_rate_change,rate_change,color='black')
    fig.savefig((os.path.join(n_path,'Rate_of_change.png')))
    
    fig, ax = plt.subplots()
    plt.xlim(0,(max(TS_graph)+10) )
    plt.ylim(0,(max(Total_Res_lib)+50))
    plt.xlabel('Time step')
    plt.ylabel('Total Resource')
    plt.plot(TS_graph,Total_Res_lib,color='black')
    fig.savefig((os.path.join(n_path,'Total Resource.png')))
    
    
    fig, ax = plt.subplots()
#    plt.ylim(0,Resource_added_per_timestep)
    plt.xlabel('Total Resource')
    plt.ylabel('Resource used per timestep')
    plt.plot(Total_res1,num_ag,color='black')
    fig.savefig((os.path.join(n_path,'Resource_per_timestep.png')))
    
#    print (reduce(lambda x, y: x + y, num_ag) / len(num_ag))
#    print (reduce(lambda x, y: x + y, Total_res1) / len(Total_res1))
    for agent in agents:
        if agent.pass_>0 and agent.extension=='N':
            print (agent.pass_, agent.extension,agent.pos,agent.need,agent.keep)
    save_data()
    movie2(agent_lib,area_max,n_path)
#    print ('1/f ave',reduce(lambda x, y: x + y, sum_keep_lib) / len(sum_keep_lib))

    return 0
if __name__ == "__main__":
    main() 