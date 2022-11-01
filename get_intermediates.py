from rdkit import Chem
from rdkit.Chem import AllChem

'''
 We provide a template for how to use the graph operation based on RDKit to search for SMILES representation of possible intermediates  in an electrocatalytic process.
 Users can modify the corresponding reactants and reaction types according to the electrocatalytic process of interest.
'''
      
           

def Iteration_add_H(origin_material):     #  Iterate from reactants
    for i in (1,2):  
        if i == 1:     # proceed the C-N coupling        
            CN_couple_product_list = CN_couple(origin_material)
            only_add_H_product = []
            drop_H2O_product = []
            for CN_couple_product in CN_couple_product_list:             
                if CN_couple_product[0] != origin_material:        
                    write_path_to_list(origin_material,only_add_H_product,drop_H2O_product,CN_couple_product[0],CN_couple_product[1])   
                    write_Intermediate_to_list([CN_couple_product[0]])   
                    #print(CN_couple_product[0])
                    Iteration_add_H(CN_couple_product[0])  
                else: 
                    continue             
        elif i == 2:     # proceed the hydrogenation reaction
            #if origin_material:
            Hydrogen = "[H]"
            react_list = [origin_material,Hydrogen]
            rxn = AllChem.ReactionFromSmarts('[*:1].[H]>>[*:1][H]')  
            reactants = [Chem.MolFromSmiles(x, sanitize=False) for x in react_list]
            products = rxn.RunReactants(reactants)
            B = [x[0] for x in list(products)]  
            #tmp_list = []     
            tmp_product = []
            drop_H2O_product = [] 
            after_drop_H2O_pro = [] 
            CN_couple_product1 = []
            N_material = [] 
            for i in B:         # boundary conditions to exclude unreasonable intermediates
                if_drop_H2O = 0  
                for atom in i.GetAtoms():
                    ele_num_error = 0            
                    if atom.GetSymbol() == "C":       
                        if atom.GetDegree() > 4:
                            ele_num_error = 1 
                            break
                        O_num = 0
                        for x in atom.GetNeighbors():      
                            if x.GetSymbol() == "H":
                                ele_num_error = 1
                                break
                            if x.GetSymbol() == "O":
                                O_num += 1
                        if O_num == 1:                   
                            for x in atom.GetNeighbors():
                                if x.GetSymbol() == "O":        
                                    for y in x.GetNeighbors():
                                        if y.GetSymbol() == "H":       
                                            ele_num_error = 1
                                            break
                                if ele_num_error:
                                    break
                        H_num = 0    
                        if O_num == 2:
                            for x in atom.GetNeighbors():          
                                if x.GetSymbol() == "O":        
                                    for y in x.GetNeighbors():
                                        if y.GetSymbol() == "H":
                                            H_num += 1
                                            break
                            if H_num == 2:
                                ele_num_error = 1
                                break
                    if ele_num_error:
                        break                                                
                    if atom.GetSymbol() == "H":        
                        if atom.GetDegree() > 1:
                            ele_num_error = 1
                            break
                    if atom.GetSymbol() == "N":        
                        if atom.GetDegree() > 3:
                            ele_num_error = 1
                            break
                        N_num = 0
                        for x in atom.GetNeighbors():          
                            if x.GetSymbol() == "H": 
                                N_num += 1
                        if N_num > 2:
                            ele_num_error = 1
                            break        
                    if atom.GetSymbol() == "O":
                        if atom.GetDegree() > 2:      
                            if_drop_H2O = 1                 
                if ele_num_error:
                    continue
                #tmp_list.append(Chem.MolToSmiles(i))
                if if_drop_H2O == 0:
                    tmp_product.append(Chem.MolToSmiles(i)) 
                if if_drop_H2O == 1:
                    drop_H2O_product.append(Chem.MolToSmiles(i))    
                    after_drop_H2O_pro = remove_H2O(drop_H2O_product)   
            #tmp_product.extend(after_drop_H2O_pro)
            #products_list = list(set(tmp_list))
            
            only_add_H_product = list(set(tmp_product))  
            drop_H2O_product = list(set(after_drop_H2O_pro)) 
            Intermediate_product = only_add_H_product + drop_H2O_product 
            write_path_to_list(origin_material,only_add_H_product,drop_H2O_product,CN_couple_product1,N_material) 
 
            if target_product not in Intermediate_product:   
                for each_product in Intermediate_product:
   
                    if each_product not in all_Intermediate:
                        Iteration_add_H(each_product)
                        all_Intermediate[each_product] = 1
            else:
                write_Intermediate_to_list(Intermediate_product)            

    #return print(Intermediate_product)

def CN_couple(origin_material):           
    #if origin_material:
    if origin_material.count("N") < 2:   
        #NO2 = "ONO"
        CN_couple_product_list = []
        N_list = ["ONO","[H]N(O)O","[H]ONO","[H]ON([H])O","[H]ON([H])O[H]","[H]NO","[H]NO[H]","[H]ON([H])[H]","[H]N","[H]N[H]","[H]N([H])O","[H]ONO[H]","NO","[H]ON","N"]      #所有可能的反应物的smile式

        for N_Intermediate in N_list:
            tmp_list = []           #
            react_list = [origin_material,N_Intermediate]     
            rxn = AllChem.ReactionFromSmarts('[*C:1].[*N:2]>>[*C:1][*N:2]')     
            reactants = [Chem.MolFromSmiles(x, sanitize=False) for x in react_list]   
            product = rxn.RunReactants(reactants)       
            B = (list(product))[0][0]     
            for atom in B.GetAtoms():
                ele_num_error = 0            
                if atom.GetSymbol() == "C":          
                    if atom.GetDegree() > 4:
                        ele_num_error = 1
                        break
            if ele_num_error: 
                tmp_list.append(origin_material)   
                tmp_list.append("")
                CN_couple_product_list.append(tmp_list)
   
            else:  
                tmp_list.append(Chem.MolToSmiles(B))
                tmp_list.append(N_Intermediate)
                CN_couple_product_list.append(tmp_list)

        return  CN_couple_product_list     
    else: 
        return [[origin_material]]


def remove_H2O(before_drop_H2O):   
    reactants = [Chem.MolFromSmiles(x, sanitize=False) for x in before_drop_H2O]
    patt = Chem.MolFromSmarts('[H]O[H]')
    after_drop_H2O = []
    for reactant in reactants:        
        product=AllChem.DeleteSubstructs(reactant,patt)
        after_drop_H2O.append(Chem.MolToSmiles(product))
    return after_drop_H2O

def write_path_to_list(origin_material,only_add_H_product,drop_H2O_product,CN_couple_product,N_material):    

    if only_add_H_product:   
        for i in only_add_H_product:
            string1 = origin_material +"  "+"+"+"  "+"H"+"  "+"-->"+"  "+ i
            #f.write('\n')
            all_path[string1] = 1

    if drop_H2O_product:   
        for i in drop_H2O_product:
            string2 = origin_material +"  "+"+"+"  "+"H"+"  "+"-->"+"  "+ i +"  "+"+"+"  "+"H2O"
            all_path[string2] = 1    

    if CN_couple_product:   

        N_reactant = N_smile_to_formula(N_material)
        string3 = origin_material +"  "+"+"+"  "+ N_reactant +"  "+"-->"+"  " + CN_couple_product
        all_path[string3] = 1

def N_smile_to_formula(N_smile):     #To convert SMILE to formula for the convenience to output the elementary step
    N_dict = {"ONO":"NO2","[H]N(O)O":"NHOO","[H]ONO":"NOOH","[H]ON([H])O":"NHOOH","[H]ON([H])O[H]":"NHOHOH","[H]NO":"NHO","[H]NO[H]":"NHOH",
    "[H]ON([H])[H]":"NH2OH","[H]N":"NH","[H]N[H]":"NH2","[H]N([H])O":"NH2O","[H]ONO[H]":"NOHOH","NO":"NO","[H]ON":"NOH","N":"N"} 
    return N_dict[N_smile]
  
    
def write_Intermediate_to_list(Intermediate_product):   
    for Intermediate_pro in Intermediate_product:     
        all_Intermediate[Intermediate_pro] = 1

def write_path(only_all_path):   
    f = open(path_file,'a')
    for path in only_all_path:
        path_list = path.split("  ")
        path_list[0] = smile_to_formula(path_list[0])
        path_list[4] = smile_to_formula(path_list[4])
        for each in path_list:
            f.write(each + "  ")
        f.write('\n')
    f.close()

def write_Intermediate(only_all_Intermediate):  
    f = open(Intermediate_file,'a')
    for Intermediate in only_all_Intermediate:    
        f.write(Intermediate)
        f.write("\n")
    f.close()     

def smile_to_formula(Intermediate):    # Convert SIMLE to formula. To facilitate analysis of the reaction elementary steps and is not necessary.
    N_num = 0 
    N_list = []
    structure = Chem.MolFromSmiles(Intermediate, sanitize=False)
    ele_num_error = 0
    for atom in structure.GetAtoms():           
        if atom.GetSymbol() == "C":
            C_O_num = 0        
            for x in atom.GetNeighbors():
                if x.GetSymbol() == "O":      
                    C_O_num += 1
            if C_O_num == 1:
                for x in atom.GetNeighbors():
                    if x.GetSymbol() == "O": 
                        for y in x.GetNeighbors():
                            if y.GetSymbol() == "H": 
                                error_Intermediate.append(Intermediate)
                                ele_num_error = 1
                                break
                    if ele_num_error:
                        break
                    else:
                        str_C = "CO"  
            if ele_num_error:
                break                        
            if C_O_num == 2:
                C_O_H_num = 0 
                for x in atom.GetNeighbors():
                    if x.GetSymbol() == "O": 
                        for y in x.GetNeighbors():
                            if y.GetSymbol() == "H":
                                C_O_H_num += 1 
                if C_O_H_num == 0:
                    str_C = "COO"
                elif C_O_H_num == 1:
                    str_C = "COOH"  
                    
        if atom.GetSymbol() == "N":
            N_num += 1
            N_H_num = 0
            N_O_num = 0
            for x in atom.GetNeighbors():     
                if x.GetSymbol() == "H":     
                    N_H_num += 1
                if x.GetSymbol() == "O":      
                    N_O_num += 1    
            if N_H_num == 0:
                str_N_H = "N"
            elif N_H_num == 1:
                str_N_H = "NH"
            elif N_H_num == 2:
                str_N_H = "NH2"
            if N_O_num == 0:
                    str_N_O = ""
            elif N_O_num == 1:
                for x in atom.GetNeighbors():
                    if x.GetSymbol() == "O": 
                        for y in x.GetNeighbors():
                            if y.GetSymbol() == "H": 
                                str_N_O = "OH" 
                                break
                            else:
                                str_N_O = "O"
            elif N_O_num == 2:
                N_O_H_num = 0
                for x in atom.GetNeighbors():
                    if x.GetSymbol() == "O": 
                        for y in x.GetNeighbors():
                            if y.GetSymbol() == "H": 
                                N_O_H_num += 1 
                if N_O_H_num == 0:
                    str_N_O = "OO"
                elif N_O_H_num == 1:
                    str_N_O = "OOH"
                elif N_O_H_num == 2:
                    str_N_O = "OHOH" 
            str_N = str_N_H + str_N_O
            N_list.append(str_N)                                                                                                                                                                                                                   
    if len(N_list)==0:
        if str_C == "COO":
            return "CO2"
        else:
            return str_C
    elif len(N_list)==1:
        return str_C + N_list[0]        
    elif len(N_list)==2:
        N_list.sort(key=len)    
        return str_C + N_list[0] + N_list[1]                          



if __name__ == '__main__':
    origin_material = "OCO"  
    target_product = "[H]N([H])C(O)N([H])[H]"     
    path_file = "./path.txt"  #  elementary step
    Intermediate_file = "./product.txt" # intermediate
    all_path = {}
    all_Intermediate = {}
    error_Intermediate = [] 
    Iteration_add_H(origin_material)
    bc = ["OCO","ONO","[H]N(O)O","[H]ONO","[H]ON([H])O","[H]ON([H])O[H]","[H]NO","[H]NO[H]","[H]ON([H])[H]","[H]N","[H]N[H]","[H]N([H])O","[H]ONO[H]","NO","[H]ON","N"]   
    for i in bc:
        all_Intermediate[i] = 1    
    only_all_path = list(all_path.keys())
    only_all_Intermediate = list(all_Intermediate.keys())
    write_path(only_all_path)
    write_Intermediate(only_all_Intermediate)

