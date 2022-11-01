import os
from package.creat_ad_structure import get_double_bond_atom,get_cross_bond_atom,get_single_bond_atom,get_bridge_bond_atom
from package.creat_ad_structure import single_ad_structure,double_ad_structure,double_cross_ad_structure
from package.get_bond_order import get_smx_bond_order
import numpy as np
from ase.io import read,write
import numpy as np
import networkx as nx
from rdkit import Chem
from package.struc_to_graph import atom_to_graph
import networkx.algorithms.isomorphism as iso
from package.default_func import get_neighbor_list,smile_to_graph,smile_to_formula,get_tri_inter_central

'''
This code is used to generate possible adsorption configutations of intermediates on nitrogen doped graphene.
The User should supply the SIMLES representation of intermediates and the slab model with atomic coordinate 
For the nitrogen-doped graphene, we considerd there possible adsorption configurations. Single adsorption, double adsorption and acr double adsorption. 
'''



obpath='/share/home/zhengss/20210412niaosu/code_up/test'  # output path
slab=read(r'/share/home/zhengss/20210412niaosu/slab/smx/POSCAR') # slab model with "POSCAR" formate 
site=[44,45] #possible adsorption site



def compare_graph(compare_structure,tmp):   #Use graph isomorphism algorithm to remove repeatedly generated structures.
    flag = 1
    neighbor_list = get_neighbor_list(tmp)
    mole_graph, chem_ad = atom_to_graph(tmp, neighbor_list,ad_atoms=[])
    bond_match = iso.categorical_edge_match('bond','')
    ads_match = iso.categorical_node_match('symbol', "")
    if compare_structure:
        for each in compare_structure:
            if iso.is_isomorphic(mole_graph,each,edge_match=bond_match,node_match=ads_match):
                flag = 0
                break
    if flag:
        compare_structure.append(mole_graph)
        return 1
    else:
        return 0            


total_count=1
smiles = ["OCN(O[H])"]   # The SIMLES representation of intermediate
for file in smiles:
    structure=[]
    compare_structure = []
    count=0
    mol = smile_to_graph(file)    
    formula=smile_to_formula(file)  
    if not os.path.exists(obpath + "/" +formula ):
        os.mkdir(obpath + "/" + formula)
    single_site=get_single_bond_atom(mol)
    double_site=get_double_bond_atom(mol)
    double_cross_site=get_cross_bond_atom(mol)
    for j in single_site:  
            flag=get_smx_bond_order(mol,mole_bond_atom=[j])
            if flag==1 :
                tmp1=single_ad_structure(slab,mol,ad_site=[site[0]],mole_bond_atom=[j])
                if compare_graph(compare_structure,tmp1):
                    structure.append(tmp1)         
    for j in single_site: 
            flag=get_smx_bond_order(mol,mole_bond_atom=[j],bond_to_basis=2)
            if flag==1 :
                tmp=single_ad_structure(slab,mol,ad_site=site,mole_bond_atom=[j])      
                if compare_graph(compare_structure,tmp):
                    structure.append(tmp)
    for j in double_site: 
            flag = get_smx_bond_order(mol,mole_bond_atom=j)
            if flag==1 :
                tmp=double_ad_structure(slab,mol,ad_site=site,mole_bond_atom=j)    
                if compare_graph(compare_structure,tmp):
                    structure.append(tmp)        
    for j in double_cross_site:
            flag = get_smx_bond_order(mol,mole_bond_atom=j)
            if flag==1 :
                tmp=double_cross_ad_structure(slab,mol,ad_site=site,mole_bond_atom=j,type="normal")       
                if compare_graph(compare_structure,tmp):
                    structure.append(tmp)       
    for i in structure:
        count+=1
        total_count+=1
        name=str(count)+"-"+formula
        write(obpath+"/"+formula+"/"+"{}.vasp".format(name),i)


