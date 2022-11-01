import os
import ase
import numpy as np
import networkx as nx
from ase.io import read,write
from ase import neighborlist
import networkx.algorithms.isomorphism as iso
from ase.data import covalent_radii
#from numba import jit
import shutil
import scipy
from ase.neighborlist import NeighborList, natural_cutoffs
from numpy.linalg import norm
from itertools import combinations
from package.default_func import get_neighbor_list


bond_match = iso.categorical_edge_match('bond','')  

ads_match = iso.categorical_node_match('symbol',"")  # example  nm = iso.categorical_edge_match(["color", "size"], ["red", 2]) 


def atom_to_graph(structure,neigh_list,ad_atoms=[]):
   
        total_graph=nx.Graph()
        for index,atom in enumerate(structure):
            #print(index)
            node_name=atom.symbol+":"+str(index)
            #node_name=index
            #print(node_name)
            total_graph.add_node(node_name,symbol=atom.symbol)
            neighbors, offsets = neigh_list.get_neighbors(index)
            #print(index,atom.symbol,neighbors)
            for i in neighbors: 
                    node_name_1=structure[i].symbol+":"+str(i)
                    if node_name!=node_name_1:
                        bond_type="".join(sorted(atom.symbol+structure[i].symbol))
                        total_graph.add_edge(node_name,node_name_1,bond=bond_type)
        #print([node for node in total_graph.nodes])
        #print([node for node in total_graph.edges])
        #print(nx.get_edge_attributes(total_graph,"bond"))

        ad_node=[]
        if len(ad_atoms)!=0:
            for i in ad_atoms:
                sym=structure[i].symbol
                ad_node.append(sym+":"+str(i))
        #print(ad_node)
        ad_graph=nx.subgraph(total_graph,ad_node)
        return total_graph,ad_graph
   
