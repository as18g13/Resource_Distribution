# -*- coding: utf-8 -*-
"""
Created on Thu Mar 22 11:27:58 2018

Irrigation algoritm which as of December 2019 has 
four different model types are introduced. These can be changed in the initial parameters
of the model.

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
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as clr
from copy import deepcopy
import time
import matplotlib as mpl
import networkx as nx
from random import randint
import pandas as pd
import csv
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.animation as animation
from sklearn import linear_model
import sklearn as skl
from scipy.stats.kde import gaussian_kde
from numpy import linspace
from scipy import optimize
from sklearn.svm import SVR 
import powerlaw as pl

#######################Parameters##############################################

r=1 #Size of neighbourhood of each agent
moore = []
Full_moore = []
neumann = [(-1,0),(0,1),(1,0),(0,-1)]
initial_coordinates = [50,50] #coordinates of initial distribution cell
arrayx=100 #Size of array Expand array size for larger maximum number of agents as there are no
arrayy=100 #periodic boundaries
area = np.zeros((arrayx,arrayy),dtype=int)
intial_previous_coordinates = -9999,-9999
Initial_agent_number = 10 #up to 5000
Initial_resource_per_agent = 50
agents = []
cells = []
Initial_Resource = Initial_agent_number*Initial_resource_per_agent
Multiple_starting_locations = 'N'
Num_simulations = 3
growth_type = 3 #Model _type 0 - LDA, 1 - LSA, 2 - GLS, 3 - GS.
Num_cells = []
Num_agents = []
TS_graph = []
Total_agents = []
agent_lib = []

min_agents = 100 #Use to change number of distribution cells in the model.
max_agents = 110 # if step_size>max_agents-min_agents, then only one simulation will be run,
step_size = 60 
size_matrix = np.arange(min_agents,max_agents,step_size) #Used ot create a matrix for multiple runs with different agent sizes. 

#########Defining Agents in system and resource flow cells###################################

class agent(): #Class which assigns agents to the system
    def __init__(self,POS,PREV,prev_agent,ID,ind_kin_group,neighbour,num,ext,p_l,cells_downstream,total_CD,time_stamp):
        self.pos = POS #Position of agent in the system
        self.prev = PREV #Position of previous agent in system
        self.prev_agent=prev_agent
        self.id = ID #ID assign to agent
        self.ind_kin_group = ind_kin_group #Individual preference 
        self.neighbours = neighbour #Neighbours of agent
        self.NH = num #Size of neighbourhood
        self.path_length = p_l
        self.extension = ext
        self.cells_downstream = cells_downstream
        self.total_CD = total_CD
        self.time_stamp = time_stamp
    
class cell():
    def __init__(self,POS,agent,ag_id):
        self.pos = POS
        self.agent = agent
        self.ag_id = ag_id
            
#############################################################################################
            
#########################Irrigation Network algoritm#########################################
       
def nrc_GS_analytical(min_agents,r): #Gives the analytical solution based on the inputed r value
    return 2*r*min_agents+(2*r+1)**2-(2*r+1)
        
def custom_moore(r): #Creates the Moore neighbourhood depending on the value r
    for i in range(-r,r+1):
        for j in range(-r,r+1):
            add = (i,j)
            if add == (0,0):
                Full_moore.append(add)
            else:
                moore.append(add)
                Full_moore.append(add)
    return moore,Full_moore
        

def check_zeros(area,x,y): #Used to calculate the number of Receiver Cells which can be added.
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
    return len(check_dic),check_dic    

def find_shortest_path_agent_can_grow(very_stoch):#Finds shortest path for each model
    keep=0 
    path_length_check = -1
    for ag in agents:
        if ag.path_length>path_length_check:
            path_length_check = ag.path_length
            keep = ag
    if very_stoch == 'Y': #if==Y then it is the GLS model
        nn = 'N'
        while (nn == 'N'): 
            ra = randint(0,(len(agents)-1))
            ag_check = agents[ra]
            nn = ag_check.extension #Agent selected at random
            keep = ag_check
    else:
        for ag in agents: #For LDA and LSA models agent is selected with lowest path length
            if ag.extension == 'Y':
                path_length_check1=ag.path_length
                if path_length_check1<path_length_check:
                    path_length_check=path_length_check1   
                    keep = ag 
    return keep

def stoch1(next_chan_sel,Nx,Ny,max_add,after_zeros,before_num): #Used for LSA model to randomise path selection
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

        
def next_channel(i_d,stoch,very_stoch,time_stamp): #Used for LDA, LSA and GLS models to find where next Distribution Cell should be added
    next_chan_sel = []
    area=update_area() #Updates the Spatial array with locations of the agents
    ag=find_shortest_path_agent_can_grow(very_stoch) #Selects agent to be tested for expansion
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
        ag.extension = 'N'
        next_channel(i_d,stoch,very_stoch,time_stamp)
    else:
        Nx,Ny=add
        path_length = ag.path_length+1
        agents.append(agent([Nx,Ny],[x,y],ag,i_d,Kin_Group_pref(),0,0,'Y',path_length,[],0,time_stamp))  
        time_stamp+=1
        ag.cells_downstream.append(agents[-1])
        for i in max_lib:
            cells.append(cell(i,(Nx,Ny),path_length))
        for c in cells:
            if c.pos==(Nx,Ny):
                cells.remove(c)   
    return time_stamp

def next_channel_max_growth(i_d,stoch,very_stoch,ags_to_expand,time_stamp): #Function used specifically for GS model
    after_zeros=0
    max_add=-9999
    add = -9999
    add_select = []
    prev_ag=[]
    for ag in ags_to_expand:
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
                            xx,yy=add
                            check_area = deepcopy(area)
                            check_area[xx][yy]=1
                            after_zeros,after_lib=check_zeros(check_area,xx,yy)
                            max_lib=after_lib
                    
                        else:
                            add = next_chan_sel[0]
                            add_select.append(add)
                            xx,yy=add
                            check_area = deepcopy(area)
                            check_area[xx][yy]=1
                            after_zeros,after_lib=check_zeros(check_area,xx,yy)
                            max_lib=after_lib
                    else:
                        if (after_zeros-before_num)>max_add:
                            max_add=(after_zeros-before_num)
                            add_select = []   
                            prev_ag = []
                            add = (Nx,Ny)
                            prev_agg = ag
                            add_select.append(add)
                            prev_ag.append(ag)
                        else:                          #comment this else out for a deterministic global selection model
                            add = (Nx,Ny)
                            prev_agg = ag
                            add_select.append(add)
                            max_add=(after_zeros-before_num)
                            prev_ag.append(ag)
                        
    if len(add_select)>1:
        aa = randint(0,(len(add_select)-1))
        add = add_select[aa]
        prev_agg = prev_ag[aa]
        xx,yy=add
        check_area = deepcopy(area)
        check_area[xx][yy]=1
        after_zeros,after_lib=check_zeros(check_area,xx,yy)  
        max_lib=after_lib
    if max_add==-9999 and add == -9999:
        ag.extension = 'N'
        ags_to_expand.remove(ag)
        next_channel_max_growth(i_d,stoch,very_stoch,ags_to_expand)
    else:
        Nx,Ny=add
        check_area = deepcopy(area)
        check_area[Nx][Ny]=1
        after_zeros,after_lib=check_zeros(check_area,Nx,Ny)  
        max_lib=after_lib
        path_length=prev_agg.path_length+1
        agents.append(agent([Nx,Ny],[x,y],prev_agg,i_d,Kin_Group_pref(),0,0,'Y',path_length,[],0,time_stamp))
        time_stamp+=1
        prev_agg.cells_downstream.append(agents[-1])
        ags_to_expand.append(agents[-1])
        for i in max_lib:
            cells.append(cell(i,(Nx,Ny),path_length))
        for c in cells:
            if c.pos==(Nx,Ny):
                cells.remove(c)   
    return ags_to_expand,time_stamp



#############################################################################################            

###############################Irrigation Area###############################################

def update_area(): #Function to create an array with the network
    area = np.zeros((arrayx,arrayy),dtype=int)
    for agent in agents:
        x,y=agent.pos
        area[x][y]=1
    for cell in cells:
        x,y=cell.pos
        area[x][y]=2
          
    return area
           
#########################Agent algorithms####################################################            

def Kin_Group_pref(): #Not used
    return float(randint(1,2))
            
def switch_growth_type(growth_type): #Function which changes the growth type based on the parameters 
    if growth_type == 0:
        stoch = 'N'
        very_stoch = 'N'
        max_growth = 'N'
    if growth_type == 1:
        stoch = 'Y'
        very_stoch = 'N'
        max_growth = 'N'   
    if growth_type == 2: 
        stoch = 'Y'
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
        
def assign_agents_flows(no_agents,path_length,time_stamp,growth_type): #
    stoch,very_stoch,max_growth = switch_growth_type(growth_type)
    No_cells = []
    ags_to_expand = []
    for i in range(no_agents): 
        print(i)
        switch_growth_type(growth_type)
        Num_cells.append(len(cells))
        Num_agents.append(len(agents))
        TS_graph.append(path_length)
        Total_agents.append((len(cells)+len(agents)))
        if i == 0: #If first cell, initial configuration is set up
            X,Y = initial_coordinates
            PrevX,PrevY = intial_previous_coordinates
            agents.append(agent([X,Y],[PrevX,PrevY],-9999,path_length,Kin_Group_pref(),0,0,'Y',0,[],0,time_stamp))
            time_stamp+=1
            ags_to_expand.append(agents[-1])
            area=update_area()
            after_zeros,after_lib=check_zeros(area,X,Y)
            for j in after_lib:
                cells.append(cell(j,(X,Y),path_length))                         
        elif max_growth == 'Y':
            ags_to_expand,time_stamp=next_channel_max_growth(i,stoch,very_stoch,ags_to_expand,time_stamp)
        else:
            time_stamp=next_channel(i,stoch,very_stoch,time_stamp)
        
        No_cells.append(len(cells))
        add_to_agent_lib(agent_lib)

    return No_cells
  
#######################Plot agents/flows########################################################

def cells_downstream(agents):
    for agent in agents:
        down_agent=agent
        upstream = down_agent.prev_agent
        while type(upstream)!=int:
            upstream.total_CD+=1
            temp = upstream.prev_agent
            upstream=temp
    downstream_data = []
    test_data = []
    for agent in agents:
        downstream_data.append(agent.total_CD)

    fig, ax = plt.subplots()
    plt.xscale('log')
    plt.hist(downstream_data,bins=np.arange(min(downstream_data), max(downstream_data) + 1, 1),edgecolor='black',facecolor='none')
    plt.xlabel('Number of Distribution Cells Downstream')
    plt.ylabel('Number of Distribution Cells')
    fig.suptitle('Histogram of the number of distribution cells downstream', fontsize=10)    
    plt.savefig('histogram.png', dpi = 300)
    plt.show()
    plt.close()

#    dd=downstream_data
    dd = np.array(downstream_data)
    dd = dd+1
    bins = int(len(dd))
    a,b = np.histogram(dd,bins = bins)
    c = a/len(dd) #normalising data to a probability
    b1 = np.delete(b,[0,0])

    #Plot of power law distribution
    c2 = []
    b2 = []
    for i in range(len(a)): #removinf zero values
        if c[i]!=0:
            c2.append(float(c[i]))
            b2.append(b1[i])

    fig, ax = plt.subplots()    
    plt.loglog(b2,c2,'k.') 

    plt.xlabel('Distribution Cells Downstream')
    plt.ylabel('Probability')
    fig.suptitle('Probability Distribution Plot for Number of Downstream Distribution Cells', fontsize=10)

    dd1 = np.array(downstream_data)
    dd1 = dd1+1
    fit1 = pl.Fit(dd1,xmin=1)
    fig1 = fit1.plot_pdf(c='black')
    fit1.power_law.plot_pdf(c='r',linestyle='-',ax=fig1)
    index1=fit1.alpha
    plt.savefig('prob_dist.png', dpi = 300)
    print('Gradient of Probability Distribution Plot: {}'.format(index1))
    plt.show()
    plt.close()

    return dd1

def Plot_agents(agents,cells): #Function to output the network created
    print(len(agents),len(cells))
    area = np.zeros((arrayx,arrayy),dtype=int)
    for agent in agents:
        x,y=agent.pos
        area[x][y]=10
    for cell in cells:
        x,y=cell.pos      
        area[x][y]=5
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
    ax.set_xlabel("Distribution Cells: {0}. Receiver Cells: {1}. r={2}".format(len(agents),len(cells),r))
    ax.imshow(area,interpolation='none',origin='lower',cmap=cmap,aspect='equal')
    fig.suptitle('Network Output', fontsize=10)

    plt.axis('on')
    fig.savefig('Netowrk_output.png', format='png', dpi=1000)
    plt.show()

    return 0 

def plot_agents1(cells):
    area = np.zeros((arrayx,arrayy),dtype=int)
    for cell in cells:
        x,y=cell.pos      
        area[x][y]=10
    number = np.amax(area)
    fig, ax = plt.subplots()
    plt.xlim(0,arrayx)
    plt.ylim(0,arrayy)
    colors = [(1.0,1.0,1.0)]
    colors.extend(mpl.cm.jet(np.linspace(0, 1, number)))
    cmap = mpl.colors.ListedColormap(colors) 
    ax.minorticks_on()
    ax.set_xticks(np.arange(-.5, arrayx, 1), minor=True)
    ax.set_yticks(np.arange(-.5, arrayy, 1), minor=True)
    ax.grid(which='minor', color='grey', linestyle='-', linewidth=0.25)
#    colors = [(1.0,1.0,1.0)]
#    colors.extend(mpl.cm.jet(np.linspace(0, 1, number)))
#    cmap = mpl.colors.ListedColormap(colors) 
#    cmap = clr.ListedColormap(['white','black','red','blue','green'])
#    bounds = [-0.5,0.5,1.5,2.5,3.5,4.5]
#    norm = clr.BoundaryNorm(bounds,cmap.N)
    ax.imshow(area,interpolation='nearest',origin='lower',cmap=cmap)
#    ax.grid(color='r', linestyle='-', linewidth=2)
#    ax.imshow(area,interpolation='nearest',origin='lower',cmap=cmap)
#    plt.scatter(coordinatesx,coordinatesy)
    plt.axis('on')
    fig.savefig('system2.png', format='png', dpi=1000)
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
    
def x_y_plot1(DC,RC):    
    fig, ax = plt.subplots()            
    plt.xlabel('Time Step')
    plt.ylabel('Number of Cells')
    timesteps = list(range(0,len(DC)))
    plt.plot(timesteps,DC,linewidth=0.5)
    plt.plot(timesteps,RC,linewidth=0.5)
    plt.savefig('check3.png', dpi = 300)
    plt.show()
    
def add_to_agent_lib(agent_lib):
    area = np.zeros((arrayx,arrayy),dtype=int)
    
    for agent in agents:
        x,y=agent.pos
        area[x][y]=20
    for cell in cells:
        x,y = cell.pos
        area[x][y]=10 
    temp = deepcopy(area)
    agent_lib.append(temp)
    return 0

def animate(i,arraylib,im,tx):
    arr = arraylib[i]
    vmax     = np.max(arr)
    vmin     = np.min(arr)
    im.set_data(arr)
    im.set_clim(vmin, vmax+1)
    tx.set_text('Timestep {0}'.format(i))

def movie2(arraylib): #Function to create animation of the network
    fig = plt.figure()
    ax = fig.add_subplot(111)
    div = make_axes_locatable(ax)
    cax = div.append_axes('right', '5%', '5%')
    cv0 = arraylib[0]
    number = np.amax(area)+10    
    colors = [(1.0,1.0,1.0)]
    colors.extend(mpl.cm.Greys(np.linspace(0, 1, number)))
    cmap = mpl.colors.ListedColormap(colors)
    tx = ax.set_title('Frame 0')
    im = ax.imshow(cv0,interpolation='nearest',origin='lower',cmap=cmap) # Here make an AxesImage rather than contour
    fig.colorbar(im, cax=cax)

    ani = animation.FuncAnimation(fig, animate, frames=len(arraylib),fargs=(arraylib,im,tx))
    ani.save('basic_animation.mp4', fps=50, extra_args=['-vcodec', 'libx264'],dpi=100)
    plt.show()
    
def plot_scaling(df):

    fig, ax = plt.subplots() 
    plt.xlabel('Number of Distribution Cells')
    plt.ylabel('Number of Branches')
    x = df['system_size']
    y = df['branches']
    max_x = max(x)
    min_x = min(x)
    X = x.iloc[:].values.reshape(-1,1)
    Y = y.iloc[:].values.reshape(-1,1)
    max_x = max(X)
    min_x = min(X)
    X_train, X_test, y_train, y_test = skl.model_selection.train_test_split(X, Y, test_size=0.2, random_state=0)
    regressor = linear_model.LinearRegression()
    regressor.fit(X_train,y_train)    
    m = regressor.coef_[0]
    b = regressor.intercept_    
    print(' y = {0} * x + {1}'.format(m, b))
    plt.scatter(x,y,c='black',s=7)
    plt.plot([min_x, max_x], [b, m*max_x + b], 'r')
    plt.savefig('branches.png', dpi = 300)
    plt.show()
    
    fig, ax = plt.subplots() 
    plt.xlabel('Number of Distribution Cells')
    plt.ylabel('Number of Receiver Cells')
    x = df['system_size']
    y = df['receiver_cells']
    max_x = max(x)
    min_x = min(x)
    X = x.iloc[:].values.reshape(-1,1)
    Y = y.iloc[:].values.reshape(-1,1)
    max_x = max(X)
    min_x = min(X)
    X_train, X_test, y_train, y_test = skl.model_selection.train_test_split(X, Y, test_size=0.2, random_state=0)
    regressor = linear_model.LinearRegression()
    regressor.fit(X_train,y_train)    
    m = regressor.coef_[0]
    b = regressor.intercept_    
    print(' y = {0} * x + {1}'.format(m, b))
    plt.scatter(x,y,c='black',s=7)
#    plt.plot(x,y,c='red')
#    plt.plot([min_x, max_x], [b, m*max_x + b], 'r')
    plt.savefig('dist_rec.png', dpi = 300)
    plt.show()
    
    fig, ax = plt.subplots() 
    plt.xlabel('Number of Distribution Cells')
    plt.ylabel('Time Taken')
    x = df['system_size']
    y = df['time_taken']

    plt.plot(x,y,c='black')
#    plt.plot(X,y_pred,c='r')
#    plt.plot([min_x, max_x], [b, m*max_x + b], 'r')
    plt.savefig('dist_time.png', dpi = 300)
    plt.show()
    
    fig, ax = plt.subplots() 
    plt.xlabel('Number of Distribution Cells')
    plt.ylabel('Number of Termini')
    x = df['system_size']
    y = df['termini']
    max_x = max(x)
    min_x = min(x)
    X = x.iloc[:].values.reshape(-1,1)
    Y = y.iloc[:].values.reshape(-1,1)
    max_x = max(X)
    min_x = min(X)
    X_train, X_test, y_train, y_test = skl.model_selection.train_test_split(X, Y, test_size=0.2, random_state=0)
    regressor = linear_model.LinearRegression()
    regressor.fit(X_train,y_train)    
    m = regressor.coef_[0]
    b = regressor.intercept_    
    print(' y = {0} * x + {1}'.format(m, b))
    plt.scatter(x,y,c='black',s=7)
    plt.plot([min_x, max_x], [b, m*max_x + b], 'r')
    plt.savefig('termini.png', dpi = 300)
    plt.show()

    fig, ax = plt.subplots() 
    plt.xlabel('Number of Distribution Cells')
    plt.ylabel('Number of Average Path Length')
    x = df['system_size']
    y = df['ave_path_length']
    max_x = max(x)
    min_x = min(x)
    X = x.iloc[:].values.reshape(-1,1)
    Y = y.iloc[:].values.reshape(-1,1)
    max_x = max(X)
    min_x = min(X)  
    X_train, X_test, y_train, y_test = skl.model_selection.train_test_split(X, Y, test_size=0.2, random_state=0)
    regressor = linear_model.LinearRegression()
    regressor.fit(X_train,y_train)    
    m = regressor.coef_[0]
    b = regressor.intercept_ 
    print(' y = {0} * x + {1}'.format(m, b))
    plt.scatter(x,y,c='black',s=7)
    plt.plot([min_x, max_x], [b, m*max_x + b], 'r')
    plt.savefig('path_length.png', dpi = 300)
    plt.show()
    
    return 0

def plot_scat(df):
    fig, ax = plt.subplots()
    plt.xlabel('Number of Distribution Cells')
    plt.ylabel('Number of Receiver Cells')
    x = df['system_size']
    y = df['receiver_cells']
    col = df['growth_type']
    plt.scatter(x,y,c=col)
    plt.savefig('distrib_rec.png', dpi = 300)
    plt.show()
    
    fig, ax = plt.subplots()
    plt.xlabel('Number of Distribution Cells')
    plt.ylabel('Time Taken (secs)')
    x = df['system_size']
    y = df['time_taken']
    col = df['growth_type']
    plt.scatter(x,y,c=col)
    plt.savefig('time_taken.png', dpi = 300)
    plt.show()
    return 0

##########################################################################################
            
def initialise(agents_flows,path_length,time_stamp,growth_type): #Sets initial conditions for simulation  
    no_agents=agents_flows
    No_cells=assign_agents_flows(no_agents,path_length,time_stamp,growth_type)
    
    return No_cells

def main():
    df = pd.DataFrame(columns=['system_size','branches','termini','ave_path_length','receiver_cells','time_taken','growth_type','model_type','tau'])
    gt_mat = []
    moore,Full_moore=custom_moore(r)
#    for gt in range(4):                        #Unblock this code to mode all growth types in one go
#        for j in range(len(size_matrix)):
#            gt_mat.append(gt)
#    df['growth_type']=gt_mat
#    for m in range(4):
#        growth_type=m
#        j = m*len(size_matrix)
#        k=j+len(size_matrix)
#        df.iloc[j:k,0]=size_matrix
    for i in range(len(size_matrix)): #This part has to be indented for all growth types too
        df['system_size']=size_matrix
        path_length = 0
        time_stamp = 1
        ag_num = size_matrix[i]
        start = time.time()
        No_cells=initialise(ag_num,path_length,time_stamp,growth_type) #Initialise the model with the number of agents and flows
        end = time.time()
        print('Time taken:{}'.format(end-start))
        print(len(agents),len(cells))
        dd=cells_downstream(agents)
        Plot_agents(agents,cells)
    #        df[i]=No_cells
        branches = 0
        termini = 0
        ave_path_length = 0.0
        total_pl = 0
        for ag in agents:
            if len(ag.cells_downstream)==0:
                termini+=1
            if len(ag.cells_downstream)>1:
                branches+=1
            total_pl+=ag.path_length
        ave_path_length=total_pl/ag_num
        rec_num = len(cells)
        time_tak = (end-start)
        j=0
        m=growth_type
        l=j+i
        df.iat[l,5]=time_tak
        df.iat[l,4]=rec_num
        df.iat[l,1]=branches
        df.iat[l,2]=termini
        df.iat[l,3]=ave_path_length
#        df.iat[l,8]=index
        if m==0:
            df.iat[l,7]= 'LDA Model'
        if m==1:
            df.iat[l,7]= 'LSA Model'
        if m==2:
            df.iat[l,7]= 'GLS Model'
        if m==3:
            df.iat[l,7]= 'GS Model'            
        del agents[:]
        del cells[:]
    num_rc=nrc_GS_analytical(min_agents,r)
    print('Total RC for Analytical GS model: {}'.format(num_rc))
#    plot_scat(df)
#        print(len(agents),len(cells))
#    x_y_plot(df,Num_simulations)
#    df=df.drop('No. Distrib Cells', 1)
#    distrib_50=(df.iloc[-1])
#    print(max(distrib_50))
#    print(min(distrib_50))
#    plot_scaling(df)
#    if Num_simulations>1:
#        pass
#    else:
#        x_y_plot1(Num_agents,Num_cells)
#        with open('num_agents_global_selection.txt', "w") as output:
#            writer = csv.writer(output)        
#           writer.writerow(['ts_graph,Num_DCs,Num_RCs,total_resource_in_cells,tot_res_in_system '])
#            writer.writerows(zip(TS_graph,Num_agents,Num_cells,Total_agents))
#        output.close() 
#    movie2(agent_lib) #Unhash to create an animation of the 
#    Plot_agents(agents,cells)
#    plot_agents1(cells)
#    plot_agents1(agents)

#    return dd
if __name__ == "__main__":
    df1=main() 