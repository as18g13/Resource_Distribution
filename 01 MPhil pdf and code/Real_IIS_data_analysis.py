# -*- coding: utf-8 -*-
"""
Created on Mon Mar 16 16:43:12 2020

@author: as18g13

Code to analyse networks when uploaded in a certain format. Created to look specifically at 
indigenous irrigation networks
"""

import pandas as pd
import networkx as nx
import numpy as np
import math as m
import matplotlib.pyplot as plt
import powerlaw as pl

nodes = []
G = nx.Graph()
posit = {}

class node():
    def __init__(self,POS,ID,upstream,downstream,total_CD):
        self.pos = POS
        self.id = ID
        self.upstream = upstream
        self.downstream = downstream
        self.total_CD = total_CD

def import_data(): #
    qanat_df = pd.read_csv('Subak_network.txt',delim_whitespace=True,skiprows=[0], \
                           header=None, names=['a','b','upstream','downstream','e'])
    return qanat_df

def clean_data(qanat_df):
    loc_data=qanat_df.e.str.split(',',expand=True)
    qanat_df=qanat_df.drop(columns=['a','b','e'])
    qanat_df['long_x']=loc_data[0]
    qanat_df['lat_y']=loc_data[1]
    
    check=qanat_df['upstream']
    for i in range(len(check)):
        b=check.iloc[i]
        check.iloc[i]=list(map(int,b.split(',')))
    qanat_df['upstream']=check
    
    check=qanat_df['downstream']
    for i in range(len(check)):
        b=check.iloc[i]
        check.iloc[i]=list(map(int,b.split(',')))
    qanat_df['downstream']=check
    
    qanat_df['lat_y']=qanat_df['lat_y'].astype(float)
    qanat_df['long_x']=qanat_df['long_x'].astype(float)    
    
    return qanat_df

def data_to_class(qanat_df):
    for i in range(len(qanat_df)):
        nodes.append(node((qanat_df['long_x'].iloc[i],qanat_df['lat_y'].iloc[i]),i+1,qanat_df['upstream'].iloc[i],qanat_df['downstream'].iloc[i],0))
    for node1 in nodes:
        up_add = []
        for up in node1.upstream:
            if up == -1:
                up_add.append(-1)
            else:
                up_ = nodes[(up-1)]
                up_add.append(up_)
        node1.upstream=up_add
        down_add = []
        for down in node1.downstream:
            if down == -1:
                down_add.append(-1)
            else:
                down_ = nodes[(down-1)]
                down_add.append(down_)
        node1.downstream=down_add
        
    return nodes

def find_limits(nodes):
    node1=nodes[0]
    long_min = node1.pos[0]
    long_max = node1.pos[0]
    lat_min = node1.pos[1]
    lat_max = node1.pos[1]
    for node1 in nodes:
        long=node1.pos[0]
        if long<long_min:
            long_min=long
        if long>long_max:
            long_max=long
        lat = node1.pos[1]
        if lat>lat_max:
            lat_max=lat
        if lat<lat_min:
            lat_min=lat
    
    long_lims=(long_min-(0.00001*long_min),long_max+(0.00001*long_max))
    lat_lims=(lat_min-abs(0.0001*lat_min),lat_max+abs(0.0001*lat_max))
    return long_lims,lat_lims

def plot_network(nodes,long_lims,lat_lims):
    G.clear()
    fig, ax = plt.subplots()
    node_list = []
    labell={}
    for node1 in nodes:
        c_point= tuple(node1.pos)
        node_list.append(c_point)
        labell[c_point]=node1.id
#        print(c_point)
        G.add_node(c_point)
        posit[c_point]=c_point
        for e in node1.upstream:
            if e==-1:
                pass
            else:
                c_edge = tuple(e.pos)
                G.add_edge(c_point,c_edge)
            
#    nx.draw(G,pos=posit,nodelist=node_list,node_size=20,node_color='black',ax=ax)
    nx.draw_networkx(G,pos=posit,nodelist=node_list,node_size=20,node_color='black',with_labels=False,ax=ax)
#    plt.xlim(0.025,0.045)
    plt.ylim(lat_lims)
    plt.xlim(long_lims)
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    fig.set_size_inches(18.5, 10.5)
    plt.savefig('network_qanat.png')

    return 0
    
def cells_downstream(nodes):
    fig, ax = plt.subplots()
    for node in nodes:
        down_agent=node
        upstream = node.upstream[0]
        while upstream!=-1:
            upstream.total_CD+=1
            temp = upstream.upstream[0]
            upstream=temp
            
    dd = []
    for node in nodes:
        dd.append(node.total_CD)
    dd1 = np.array(dd)

    bins = int(len(dd))
    a,b = np.histogram(dd,bins = bins)
    c = a/len(dd) #normalising data to a probability
    b1 = np.delete(b,[0,0])
    fig, ax = plt.subplots()
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

    dd1 = dd1+1
    fit1 = pl.Fit(dd1)
    fig1 = fit1.plot_pdf(c='black')
    fit1.power_law.plot_pdf(c='r',linestyle='-',ax=fig1)
    index1=fit1.alpha
    print(fit1.power_law.KS())
    plt.savefig('prob_dist_qanat.png', dpi = 300)
    print(index1)
    plt.show()
    
    fig, ax = plt.subplots()
    plt.xscale('log')
    plt.hist(dd,bins=np.arange(min(dd), max(dd) + 1, 1),edgecolor='black',facecolor='none')
    plt.xlabel('Number of Distribution Cells Downstream')
    plt.ylabel('Number of Distribution Cells')
    plt.savefig('histo_qanat.png', dpi = 300)
    plt.show()

    return 0

def main():
    qanat_df=import_data()
    qanat_df=clean_data(qanat_df)
    nodes=data_to_class(qanat_df)
    long_lims,lat_lims=find_limits(nodes)
    plot_network(nodes,long_lims,lat_lims)
    cells_downstream(nodes)
    return nodes

qanat_df=main()