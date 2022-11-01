import numpy as np
import os
import math
import networkx as nx
from ase.io import read
from ase.io import write
from ase import neighborlist
import networkx.algorithms.isomorphism as iso
from ase.data import covalent_radii
import shutil
from numpy.linalg import norm
from itertools import combinations
from package.mole_operation import dou_ad_entension,sin_ad_extension,sin_neg_extension,mole_extension, get_base_position
from package.default_func import get_neighbor_list,periodic_deal,get_radii
from ase import Atom,Atoms

'''
This part of the codes contain the enumeration of possible adatoms in molecular graphs and the use of geometric rules and graph connectivity to construct 3D adsorption configurations
The geometric rules are partial borrow from the Catkit (J. Phys. Chem. A 2019, 123, 2281 - 2285).
'''

def get_single_bond_atom(mol):   #enumerate the atoms adsorbed on a single active site 
    element=nx.get_node_attributes(mol,"symbol")
    C_N_list = []
    O_list = []              
    mole_bond_atom=[]      
    for key,value in element.items():
        if value == "C" or value == "N":
            C_N_list.append(key)
        if value == "O":
            O_list.append(key)
    for index in C_N_list:
        if mol.degree(index) <= 3:
            mole_bond_atom.append(index)
    for index in O_list:
        if mol.degree(index) <= 1:
            mole_bond_atom.append(index)  
    return mole_bond_atom  

def get_bridge_bond_atom(mol):  #enumerate the atoms adsorbed on the bridge site  
    element=nx.get_node_attributes(mol,"symbol")
    C_N_list = []
    O_list = []              
    mole_bond_atom=[]      

    for key,value in element.items():
        if value == "C" or value == "N":
            C_N_list.append(key)
        if value == "O":
            O_list.append(key)
    for index in C_N_list:
        if mol.degree(index) <= 2:
            mole_bond_atom.append(index)
    for index in O_list:
        if mol.degree(index) <= 0:
            mole_bond_atom.append(index)  
    return mole_bond_atom  

def get_double_bond_atom(mol):    #enumerate the atom pairs that can co-adsorbed on the two active sites by "double adsorption mode"  
    element=nx.get_node_attributes(mol,"symbol")
    C_N_list = []
    O_list = []
    mole_bond_atom=[]
    for key,value in element.items():
        if value == "C" or value == "N":
            C_N_list.append(key)
        if value == "O":
            O_list.append(key)    
    for index1 in C_N_list:
        if mol.degree(index1) <= 3:
            for index2 in C_N_list:
                if index2 != index1 and mol.degree(index2) <= 3:
                    if ((index1,index2) in mol.edges) or ((index2,index1) in mol.edges): 
                        mole_bond_atom.append([index1,index2])
            for index2 in O_list:
                if mol.degree(index2) <= 1:
                    if ((index1,index2) in mol.edges) or ((index2,index1) in mol.edges): 
                        mole_bond_atom.append([index1,index2])
                        mole_bond_atom.append([index2,index1])              
    return mole_bond_atom

def get_cross_bond_atom(mol):     #enumerate the atom pairs that can co-adsorbed on  two active sites by "arc double adsorption mode"  
    element=nx.get_node_attributes(mol,"symbol")
    C_N_list = []
    O_list = []
    mole_bond_atom=[]
    for key,value in element.items():
        if value == "C" or value == "N":
            C_N_list.append(key)
        if value == "O":
            O_list.append(key) 
    for index1 in C_N_list:
        if mol.degree(index1) <= 3:
            index1_neighbor = mol[index1]
            for key, _ in index1_neighbor.items():
                if mol.degree(key) > 1:
                    index1_neighbor_2 = mol[key]
                    for index2, _ in index1_neighbor_2.items():
                        if index2 != index1:
                            if element[index2]=="C" or element[index2]=="N":
                                if mol.degree(index2) <= 3 and ([index2,index1] not in mole_bond_atom):
                                    mole_bond_atom.append([index1,index2])
                            if element[index2]=="O":
                                if mol.degree(index2) <= 1:   
                                    mole_bond_atom.append([index1,index2])  
    for index1 in O_list:                                
        if mol.degree(index1) == 1:
            index1_neighbor = mol[index1]
            for key, _ in index1_neighbor.items():
                if mol.degree(key) > 1:
                    index1_neighbor_2 = mol[key]
                    for index2, _ in index1_neighbor_2.items():
                        if index2 != index1:
                            if element[index2]=="O":
                                if mol.degree(index2) <= 1 and ([index2,index1] not in mole_bond_atom):   
                                    mole_bond_atom.append([index1,index2])  
                             
    return mole_bond_atom


def single_ad_structure(slab,mol,ad_site,mole_bond_atom=[],type="auto"):  # enumerate the possible adsorption configurations of intermediates that adsorb by "single adsorption mode"
    element=nx.get_node_attributes(mol,"symbol")
    slab=slab.copy()  
    new_mol = mol.copy()
    positions = {}
    for i in list(new_mol.nodes):   
        positions[i]=(0,0,0)
    #print("element",element)
    if type=="auto":
        if len(ad_site)==1:  # single site
            print(element[0])
            R=get_radii(slab[ad_site[0]].symbol)+get_radii(element[mole_bond_atom[0]])
            pos_ad_site_neigh=[slab[i].position for i in ad_site ]
    
            positions[mole_bond_atom[0]]=tuple(get_base_position(pos_ad_site_neigh,[R]))
        elif len(ad_site)==2:  # bridge site
 
            pos_ad_site_neigh=[slab[i].position for i in ad_site ]
            a=np.linalg.norm(slab[ad_site[0]].position-slab[ad_site[1]].position)
            R1=get_radii(slab[ad_site[0]].symbol)
            R2=get_radii(slab[ad_site[1]].symbol)
            R3=get_radii(element[mole_bond_atom[0]])
            R=[(R2+R3),(R1+R3)]
            positions[mole_bond_atom[0]]= tuple(get_base_position(pos_ad_site_neigh,R))   
    else:
        R=-1  #  user defined adsorption height. 
        positions[mole_bond_atom[0]]=tuple(slab[ad_site[0]].position+np.array([0,0,R]))

    total_branches = list(nx.bfs_successors(new_mol, mole_bond_atom[0])) 
    if len(total_branches[0][1]) != 0:  
        sin_ad_extension(positions, element, total_branches[0])  
        center_atom = total_branches[0][0]
        graph_copy=new_mol.copy()        
        if len(total_branches[1:]) > 0: 
            if len(total_branches[1][1]) == 2:   
                graph_copy.remove_edge(*[total_branches[1][0],total_branches[0][0]])
                bran_tmp0=list(nx.bfs_successors(graph_copy, total_branches[1][1][0]))
                bran_tmp1 = list(nx.bfs_successors(graph_copy, total_branches[1][1][1]))               
        for branch in total_branches[1:]: 
            mole_extension(positions,element, branch, center_atom) 
    atom_list=[] 
    for i in list(new_mol.nodes):
        a = Atom(element[i],positions[i])
        atom_list.append(a)
    atoms = Atoms(atom_list)
    slab += atoms         
    return slab


def double_ad_structure(slab,mol,ad_site,mole_bond_atom=[],type="auto"):# enumerate the possible adsorption configurations of intermediates that adsorb by "double adsorption mode"
    element=nx.get_node_attributes(mol,"symbol")
    new_mol = mol.copy()  
    slab = slab.copy()   
    positions = {}    
    for i in list(new_mol.nodes):  
        positions[i]=(0,0,0)
    r_bond=[]
    r_site=[]
    
    for i in mole_bond_atom:
        r_bond.append(get_radii(element[i]))

    for i in ad_site:
        r_site.append(get_radii(slab[i].symbol))

    if type=="auto":
            new_site_a=get_base_position([slab[ad_site[0]].position],[(r_site[0]+r_bond[0])])
            new_site_b=get_base_position([slab[ad_site[1]].position],[(r_site[1]+r_bond[1])])
    else:
            H=1.5  #  user defined adsorption height. 
            new_site_a = slab[ad_site[0]].position + np.array([0, 0, H])
            new_site_b = slab[ad_site[1]].position + np.array([0, 0, H])   
    vec_site = new_site_b - new_site_a  
    len_vec = np.linalg.norm(vec_site)  
    uvec0 = vec_site / len_vec  
    d = np.sum(r_bond)
    dn = (d - len_vec) / 2  
    base_position0 = new_site_a - uvec0 * dn    
    base_position1 = new_site_b + uvec0 * dn        
    positions[mole_bond_atom[0]] = tuple(base_position0)
    positions[mole_bond_atom[1]] = tuple(base_position1)
    if tuple(mole_bond_atom) in new_mol.edges:
            new_mol.remove_edge(*mole_bond_atom)

    uvec1  = np.array([[0,0,1]])  
    uvec2 = np.cross(uvec1, uvec0)      
    uvec2 = uvec2/np.linalg.norm(uvec2)
    uvec1 = -np.cross(uvec2, uvec0) 
    #print("uc1", uvec1)
    

    branches0 = list(nx.bfs_successors(new_mol, mole_bond_atom[0]))
    if len(branches0[0][1]) != 0:
            uvec = [-uvec0, uvec1[0], uvec2[0]]  
            dou_ad_entension(positions,element, uvec, branches0[0])
            root = branches0[0][0]

            graph_copy = new_mol.copy()
            
            if len(branches0[1:])>0:
                if len(branches0[1][1]) == 2:
                    graph_copy.remove_edge(*[branches0[1][0], branches0[0][0]])
                    bran_tmp0 = list(nx.bfs_successors(graph_copy, branches0[1][1][0]))
                    bran_tmp1 = list(nx.bfs_successors(graph_copy, branches0[1][1][1]))
            for branch in branches0[1:]:
                mole_extension(positions,element, branch, root)


    branches1 = list(nx.bfs_successors(new_mol, mole_bond_atom[1]))
    if len(branches1[0][1]) != 0:
            uvec = [uvec0, uvec1[0], uvec2[0]]
            dou_ad_entension(positions,element, uvec, branches1[0])
            root = branches1[0][0]

            graph_copy = new_mol.copy()
            
            if len(branches1[1:]) > 0:
                if len(branches1[1][1]) == 2:
                    graph_copy.remove_edge(*[branches1[1][0], branches1[0][0]])
                    bran_tmp0 = list(nx.bfs_successors(graph_copy, branches1[1][1][0]))
                    bran_tmp1 = list(nx.bfs_successors(graph_copy, branches1[1][1][1]))
            for branch in branches1[1:]:
                mole_extension(positions,element, branch, root)

        
    atom_list=[]
    for i in list(new_mol.nodes):
            a = Atom(element[i],positions[i])  
            atom_list.append(a)
    atoms = Atoms(atom_list)    
    slab += atoms
    return slab

def double_cross_ad_structure(slab,mol,ad_site,mole_bond_atom=[],type="auto"): # enumerate the possible adsorption configurations of intermediates that adsorb by " arc double adsorption mode"
    element=nx.get_node_attributes(mol,"symbol")
    slab = slab.copy()
    new_mol = mol.copy()
    positions = {}
    for i in list(new_mol.nodes):   #初始化坐标信息
        positions[i]=(0,0,0)    
    vec_site_0=slab[ad_site[0]].position-slab[ad_site[1]].position

    r_bond=[]
    r_site=[]
    
    for i in mole_bond_atom:
        r_bond.append(get_radii(element[i]))

    for i in ad_site:
        r_site.append(get_radii(slab[i].symbol))

    inter_atom_index = 0
    connec_list_of_bond = []
    for i in mole_bond_atom:
        tmp = []
        for j in new_mol.edges:
            if i in j:
                tmp.append(j[0])
                tmp.append(j[1])
        connec_list_of_bond.append(tmp)
    for i in connec_list_of_bond[0]:
        for j in connec_list_of_bond[1]:
            if i == j:
                inter_atom_index = i
    
    if type=="auto":
            new_site_a=get_base_position([slab[ad_site[0]].position],[(r_site[0]+r_bond[0])])
            new_site_b=get_base_position([slab[ad_site[1]].position],[(r_site[1]+r_bond[1])])
    else:
            H=1.5  
            new_site_a = slab[ad_site[0]].position + np.array([0, 0, H])
            new_site_b = slab[ad_site[1]].position + np.array([0, 0, H])    
    vec_site = new_site_b- new_site_a  
    len_vec = np.linalg.norm(vec_site)  
    uvec0 = vec_site / len_vec 



    d = 2.1  # user defined
    dn = (d - len_vec) / 2  
    base_position0 = new_site_a - uvec0 * dn
    base_position1 = new_site_b + uvec0 * dn
    positions[mole_bond_atom[0]] = tuple(base_position0)
    positions[mole_bond_atom[1]] = tuple(base_position1)
    
    uvec1 = np.array([[0, 0, 1]])
    uvec2 = np.cross(uvec1, uvec0)  
    uvec2 = uvec2/np.linalg.norm(uvec2)  

    b=r_bond[0]+get_radii(element[inter_atom_index])
    c=r_bond[1]+get_radii(element[inter_atom_index])
    positions[inter_atom_index]= get_base_position([base_position1,base_position0],[b,c])  
    new_mol.remove_edge(*[mole_bond_atom[0],inter_atom_index])
    new_mol.remove_edge(*[mole_bond_atom[1], inter_atom_index])
    uvec1=np.cross(uvec0,uvec2)
    uvec1 = uvec1/np.linalg.norm(uvec1)  

    branches0 = list(nx.bfs_successors(new_mol, mole_bond_atom[0]))
    if len(branches0[0][1]) != 0:
            uvec = [-uvec0, uvec1[0], uvec2[0]]  
            
            dou_ad_entension(positions,element, uvec, branches0[0])
            root = branches0[0][0]
            for branch in branches0[1:]:
                mole_extension(positions,element, branch, root)      

    branches1 = list(nx.bfs_successors(new_mol, mole_bond_atom[1]))
    if len(branches1[0][1]) != 0:
            uvec = [uvec0, uvec1[0], uvec2[0]]
            dou_ad_entension(positions,element, uvec, branches1[0])
            root = branches1[0][0]
            for branch in branches1[1:]:
                mole_extension(positions,element, branch, root)
   

    branches2=list(nx.bfs_successors(new_mol, inter_atom_index))
    if len(branches2[0][1]) != 0:
            sin_ad_extension(positions,element, branches2[0],cross="y",vec=uvec2[0])
            root = branches2[0][0]
            for branch in branches2[1:]:
                mole_extension(positions,element, branch, root)

    
    atom_list=[]
    for i in list(new_mol.nodes):
            a = Atom(element[i],list(positions[i]))
            atom_list.append(a)
    atoms = Atoms(atom_list)      
    slab += atoms
    return slab

