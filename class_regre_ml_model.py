import numpy as np
import os
import shutil
import sys
import time
import warnings
from random import sample
from ase import Atoms
import scipy
from ase.neighborlist import NeighborList, natural_cutoffs
from itertools import combinations
from package.default_func import * 
import networkx as nx
import networkx.algorithms.isomorphism as iso
from ase.neighborlist import NeighborList, natural_cutoffs
from ase.io import read,write
from package.struc_to_graph import atom_to_graph
import networkx.algorithms.isomorphism as iso
from torch import optim
from get_reaction_network import *
import sklearn
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import datasets,metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
import random
from get_reaction_network import get_struc,get_ML_reaction_network,get_gasfile

'''
 The active learning algorithm for the prediction of the most likely reaction pathway
 This algorithm includes teo part: the stability prediction by GradientBoostingClassifier and the formation energy prediction by Ridge regression
'''




bond_match = iso.categorical_edge_match('bond','')  

ads_match = iso.categorical_node_match('symbol',"")
c_form=-8.15
o_form=-7.67
h_form=-3.46
n_form=-3.62
slab_dic={}

def get_formation_e(i):   
    e=float(i[5])
    basis_h_num=[node for node in i[2] if node[0]=="H"]
    hcount=ocount=ccount=ncount=0
    for node in i[1].nodes:
        if node[0]=="H":
            hcount+=1
        if node[0]=="C":
            ccount+=1
        if node[0]=="O":
            ocount+=1
        if node[0]=="N":
            ncount+=1
    formation_e=e-slab_dic[1]-(hcount)*h_form-ccount*c_form-ocount*o_form-ncount*n_form
    #print(i[4],formation_e)
    return formation_e




def get_regre_data(total_data):   #  Convert the CONTCAR format of material to the feature for formation energy prediction model
    x_data=[]
    y_data=[]
    name_data=[]
    for file in os.listdir(total_data):

        for sub in os.listdir(total_data+"/"+file):
            if ".vasp" in sub :
                struc=read(total_data+"/"+file+"/"+sub)
                struc_ensem=get_struc_real(struc,file)
                energy=0  
                for line in open(total_data+"/"+file+"/scf/output"):
                    if "1 F" in line:
                        energy=float(line.split()[4])

                thermal_correction=get_thermal_correction_from_vaspkit_output(total_data+"/"+file)
                free_energy=energy+thermal_correction
                struc_ensem.append(free_energy)
                fea_zero=np.zeros(32)
                for node in struc_ensem[1]:   
                    nei_tmp=list(struc_ensem[1].neighbors(node))
                    nei_tmp.append(node)
                    frag_tmp="".join(sorted([i[0] for i in nei_tmp]))
                    for i in frag_list:
                        if i ==frag_tmp:
                            index=frag_list.index(i)
                            fea_zero[index]+=1
                fea_mat=fea_zero
                formation_e=get_formation_e(struc_ensem)
                y_data.append(formation_e)
                x_data.append(fea_mat)
                name_data.append(file)

    #print(len(x_data))
    x_data=np.array(x_data)
    y_data=np.array(y_data)
    return x_data,y_data,name_data
def get_classfier_data(total_data): #  Convert the CONTCAR format of material to the feature for formation stability model
    x_data=[]
    y_data=[]
    name_data=[]
    for file in os.listdir(total_data):
        for sub in os.listdir(total_data+"/"+file):
            if ".vasp" in sub :
                struc=read(total_data+"/"+file+"/"+sub)
                struc_ensem=get_struc_real(struc,file)
                struc1=read(total_data+"/"+file+"/CONTCAR")
                struc_ensem_1=get_struc_real(struc1,file)
                fea_zero=np.zeros(32) 
                for node in struc_ensem[1]:   
                    nei_tmp=list(struc_ensem[1].neighbors(node))
                    nei_tmp.append(node)
                    frag_tmp="".join(sorted([i[0] for i in nei_tmp]))
                    for i in frag_list:
                        if i ==frag_tmp:
                            index=frag_list.index(i)
                            fea_zero[index]+=1
                fea_mat=fea_zero  
                flag=1
                if nx.is_isomorphic(struc_ensem_1[1],struc_ensem[1],edge_match=bond_match) :
                            #formation_e=get_formation_e(i)
                            y_data.append(1)
                            x_data.append(fea_mat)
                            flag=0
                            name_data.append(file)
                if flag==1:
                        y_data.append(-1)
                        x_data.append(fea_mat)
                        name_data.append(file)
    x_data=np.array(x_data)
    y_data=np.array(y_data)
    return x_data,y_data,name_data
def regre_model(x_train,y_train,x_test):  #  the regre_model
    clf=Ridge(alpha=1).fit(x_train,y_train)
    value=clf.predict(x_test)
    return value
def classfy_model(x_train,y_train,x_test): #  the classfy model
        clf = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1,max_depth=3, random_state=0).fit(x_train, y_train) 
        value=clf.predict(x_test)
        return value

def get_struc_ensemble(name_energy_dic):
    struc_total=[]
    for i in name_energy_dic.keys():  #
        struc=read(total_data_bf+"/"+i+"/"+i+".vasp")
        tmp=get_struc(struc,i)
        tmp.append(name_energy_dic[i])
        #print(tmp)
        struc_total.append(tmp)
    return struc_total

def get_path_inter(path):
    inter_list=[]
    #path_tmp=path[1]
    for i in path:
        for j in i:
            if j not in inter_list:
                inter_list.append(j)
    return inter_list


def get_gzy_path_no_inter(nh2_path,det_step):    #  To identify the most likely reaction pathway from the predicted reaction network
    count=0
    min_ini=2  
    min_total_e_ini=18
    ob_path=''
    for n_path in enumerate(nh2_path):
        n_path=list(n_path)
        reaction_energy=[]
        flag=0
        for i in n_path[1]:
            tmpe=energydict[i]
            tmptypr=typedict[i]
            reaction_energy.append(float(tmpe))               
        n_path.append(reaction_energy)
        e_path=float(max(n_path[2]))
        e_positive=[i for i in n_path[2] if i>0]
        e_total_path=float(sum(e_positive))
        if e_path<min_ini or (e_path==min_ini and e_total_path<min_total_e_ini):
            min_ini=e_path
            ob_path=n_path
            min_total_e_ini=e_total_path

        #total_reaction_route.append(n_path)
        count+=1
    if ob_path=='':
        conver="no path"
        inter_list=[]
        return conver,inter_list
    else: 
        pre_step_e=dict(sorted(dict(zip(ob_path[1],ob_path[2])).items(),key=lambda x:x[1],reverse=True))  
        positive_intet=[]
        positive_e=[]
        for i,j in pre_step_e.items():
            if j >0:
                positive_intet.append(i)
                positive_e.append(j)
        print("reee",positive_e,sum(positive_e), positive_intet,[i for i in det_step.keys()])
        #positive_intet=sorted(positive_intet)
        if positive_intet==[i for i in det_step.keys()]:  
            conver="reach object"
            inter_list=get_path_inter(ob_path[1])  
            return conver,inter_list
        else:
            conver="do not reach object"          
            inter_list=get_path_inter(ob_path[1])
            return conver,inter_list
                   
      
    

def graph_degree_add(train_index,test_index,reaction_network,num_step):  #   The degree anslysis of the reaction network and calculate the intermediates with high degree for the next iteration. 
    node_list=[node for node in reaction_network.nodes()]
    node_degree={}
    for i in node_list:
        tmp_de=reaction_network.degree(i)
        node_degree[i]=tmp_de
    new_node_degree=dict(sorted(node_degree.items(),key=lambda x:x[1],reverse=True))
    new_node=[i for i in new_node_degree.keys()]
    #print("kankan",len(new_node))
    count=0
    for i in new_node:
                tmp_index=class_name_data.index(i)
                if count<num_step:
                    if tmp_index not in train_index :
                        train_index.append(tmp_index)
                        test_index.remove(tmp_index)
                        count+=1    
    if count<num_step:
          train_app_index=random.sample(test_index,num_step-count)
          train_index=train_index+train_app_index
          test_index=[i for i in test_index if not i in train_app_index]
    print("train_test_len",len(train_index),len(test_index))
    return train_index,test_index

def path_random_add(train_index,test_index,reaction_network,inter_list):
                app_index=[]
                for i in inter_list:
                    if i not in x_train_name:
                        tmp_index=class_name_data.index(i)
                        app_index.append(tmp_index)
                inter_num=len(app_index)
                train_index_tmp=train_index+app_index
                #print("1",len(train_index_tmp))
                test_index_tmp=[i for i in data_index if not i in train_index_tmp]
           
                train_index=train_index_tmp
                test_index=test_index_tmp
                    #test_index=[i for i in data_index if not i in train_index]
                    #print("new1",len(train_index),len(test_index))   
                return train_index,test_index,inter_num 


total_data=r''
total_data_bf=r''
base_path=r''
relax_path=r''
gaspath=r''
#relax_path=r''
s = time.time()

bond_match = iso.categorical_edge_match('bond','')  

ads_match = iso.categorical_node_match('symbol',"")  # example  nm = iso.categorical_edge_match(["color", "size"], ["red", 2]) 

gas=get_gasfile(gaspath)
 
zy_det_step={}  

# the possible frag of the intermediates
frag_list=[]


x_ini_class_data,y_ini_class_data,class_name_data=get_classfier_data(total_data_bf) 
x_ini_reg_data,y_ini_reg_data,reg_name_data=get_regre_data(relax_path)          


data_index=[i for i in range(0,len(class_name_data))]
reactant_and_product=[i for i in range(0,len(class_name_data)) if class_name_data[i]==" " or class_name_data[i]==" "]  #give the reactant and product
print("rrrres",reactant_and_product)
num_step=len(data_index)//10
total_num=[]
dingwei=0
for test_num in range(0,100):
    dingwei+=1
    print("iteration loop",dingwei)
    train_index=random.sample(data_index,num_step)
    for i in reactant_and_product:
        if i not in train_index:
            train_index.append(i)  
    test_index=[i for i in data_index if not i in train_index]
    print("ini",len(train_index),len(test_index))
    loop=0
    while loop<14:  # if the loop exceed 14, it means the model fail.
        x_train_name=[class_name_data[i] for i in train_index]  
        x_train_class=x_ini_class_data[train_index]   
        y_train_class=y_ini_class_data[train_index]  
        y_train_name=[ class_name_data[i] for i in train_index] 
        x_test_class=x_ini_class_data[test_index]
        x_test_name=[ class_name_data[i] for i in test_index]
        value_class=classfy_model(x_train_class,y_train_class,x_test_class)  
        num_1_list_test=[]  
        num_1_list_tra=[]   
        num_1_test_name=[]  
        num_1_tra_name=[] 
        num_neg_1_list_tra=[] 
        num_neg_1_list_name=[] 
        x_test_regre=[]

        for i in  [i[0] for i in enumerate(value_class.tolist()) if i[1]==1 ]:            
                num_1_list_test.append(x_test_name[i])  
     

        y_train_1=[i[0] for i in enumerate(y_train_class.tolist()) if i[1]==1 ]     
        for k in y_train_1:  
                num_1_list_tra.append(y_train_name[k])
        y_train_neg_1=[i[0] for i in enumerate(y_train_class.tolist()) if i[1]==-1] # 
        for k in y_train_neg_1:  
                num_neg_1_list_tra.append(y_train_name[k])
        x_train_regre=[]
        y_train_regre=[]
        tmp=[]
        for i in num_1_list_tra:
            tmp.append(i)
            for j in reg_name_data:
                if j==i:
                    tmp_index1=reg_name_data.index(i)
                    x_train_regre.append(x_ini_reg_data[tmp_index1])
                    y_train_regre.append(y_ini_reg_data[tmp_index1])
                    num_1_tra_name.append(reg_name_data[tmp_index1])  
        for i in num_neg_1_list_tra:  # use graph isomorphism to check if the relaxed structure is in the unrelaxed ensemble. 
                atom=read(total_data_bf+"/"+i+"/CONTCAR")
                neighbor_list=get_neighbor_list(atom,cutoff=1.25)
                full,chem=atom_to_graph(atom,neighbor_list,ad_atoms=[])
                #tmp=[subfile,full]
                flag=1
                for j in num_1_list_test:    
                    atom1=read(total_data_bf+"/"+j+"/POSCAR")
                    neighbor_list1=get_neighbor_list(atom1,cutoff=1.25)
                    full1,chem1=atom_to_graph(atom1,neighbor_list1,ad_atoms=[])
                    if nx.is_isomorphic(full,full1,edge_match=bond_match,node_match=ads_match):   
                                flag+=1
                                if j in reg_name_data:
                                    tmp_index1=reg_name_data.index(j)
                                    x_train_regre.append(x_ini_reg_data[tmp_index1])
                                    y_train_regre.append(y_ini_reg_data[tmp_index1])
                                    num_1_tra_name.append(reg_name_data[tmp_index1])
                                    num_1_list_test.remove(j)
 
                                break
        for i in num_1_list_test:
            x_test_regre.append(x_ini_class_data[class_name_data.index(i)]) 
        print("Regre",len(x_train_regre),len(x_test_regre))
        if len(x_test_regre)!=0:    
            value_regre=regre_model(x_train_regre,y_train_regre,x_test_regre) 
            reg_test_name_energy=dict(zip(num_1_list_test,value_regre))
            reg_tra_name_energy=dict(zip(num_1_tra_name,y_train_regre))
            reg_test_str=get_struc_ensemble(reg_test_name_energy)  
            reg_train_str=get_struc_ensemble(reg_tra_name_energy)
            total_str=reg_train_str+reg_test_str   
            reaction_network=get_ML_reaction_network(total_str,gas)  
            energydict=nx.get_edge_attributes(reaction_network,"deltE")
            typedict=nx.get_edge_attributes(reaction_network,"type")
            total_reaction_route=[]
            nh2_path=nx.all_simple_edge_paths(reaction_network,"1_NO2",'1_CONH2NH2')       
            conver_flag,inter_list=get_gzy_path_no_inter(nh2_path,zy_det_step) 
            if conver_flag=="reach object" :
                    print("reach_________________________________________________")
                    t1 = time.time()
                    total_num.append(len(y_train_class))
                    print("inter",inter_list)
                    print("time--", int(t1 - s))
                    break
            else:  
                if len(train_index)>=(len(data_index)-num_step):
                    total_num.append(len(y_train_class))
                    t1 = time.time()
                    print("fail--", int(t1 - s))
                    break
                else:
                    if len(inter_list)==0:
                        train_index,test_index=graph_degree_add(train_index,test_index,reaction_network,num_step)
                        loop+=1
                    else:
                        train_index,test_index,inter_num=path_random_add(train_index,test_index,reaction_network,inter_list)
                        tmp_num=num_step-inter_num
                        #print("pp",tmp_num)
                        train_index,test_index=graph_degree_add(train_index,test_index,reaction_network,tmp_num)
                        loop+=1
        else:  #  if not intermediates need be predicted while no converage.
            reg_tra_name_energy=dict(zip(num_1_tra_name,y_train_regre))
            #print(reg_test_name_energy,reg_tra_name_energy)
            reg_train_str=get_struc_ensemble(reg_tra_name_energy)
            total_str=reg_train_str  
            reaction_network=get_ML_reaction_network(total_str,gas)  
            energydict=nx.get_edge_attributes(reaction_network,"deltE")
            typedict=nx.get_edge_attributes(reaction_network,"type")
            total_reaction_route=[]
            nh2_path=nx.all_simple_edge_paths(reaction_network,"1_NO2",'1_CONH2NH2')
            conver_flag,inter_list=get_gzy_path_no_inter(nh2_path,zy_det_step)       
            if conver_flag=="reach object" :
                    print("reach the real pathway")
                    t1 = time.time()
                    total_num.append(len(y_train_class)) 
                    print("inter",inter_list)
                    print("time--", int(t1 - s))
                    break
            else:                
                if len(train_index)>=(len(data_index)-num_step):
                    total_num.append(len(y_train_class))
                    t1 = time.time()
                    print("time--", int(t1 - s))
                    break
                else:
                    if len(inter_list)==0: 
                        train_index,test_index=graph_degree_add(train_index,test_index,reaction_network,num_step)
                        loop+=1
                    else:
                        train_index,test_index,inter_num=path_random_add(train_index,test_index,reaction_network,inter_list)
                        tmp_num=num_step-inter_num
                        train_index,test_index=graph_degree_add(train_index,test_index,reaction_network,tmp_num)
                        loop+=1

                    

print("reach_num",total_num)
d=[i/len(data_index)*100 for i in total_num]
d=np.array(d)
mean_v=np.mean(d)
std_v=np.std(d,ddof=1)
print(mean_v,std_v)
    


        


