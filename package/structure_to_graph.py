import os
import ase
import numpy as np
import networkx as nx
from ase.io import read
from ase.io import write
from ase import neighborlist
import networkx.algorithms.isomorphism as iso
from ase.data import covalent_radii
#from numba import jit
import shutil
import scipy
from numpy.linalg import norm
from itertools import combinations
from package.default_func import get_neighbor_list


bond_match = iso.categorical_edge_match('bond','')  #graph要比较的属性，属性的默认值

# Handles isomorphism for atoms with regards to perodic boundary conditions
ads_match = iso.categorical_node_match(['symbol'], [-1, False])

def normalize(vector):
    return vector / norm(vector) if norm(vector) != 0 else vector * 0

def offset_position(atoms, neighbor, offset):
   #print("offset_pos",atoms[neighbor].position + np.dot(offset, atoms.get_cell()))
   return atoms[neighbor].position + np.dot(offset, atoms.get_cell())   #The positions of the neighbor atoms，offset应该是处理跨过边界的问题

def connected_component_subgraphs(G):  #获得连通子图，这是什么意思？
    for c in nx.connected_components(G):
        yield G.subgraph(c)

def bond_symbol(atoms, a1, a2):
    return "{}{}".format(*sorted((atoms[a1].symbol, atoms[a2].symbol)))

def node_symbol(atom, offset):   #获得某个原子的符号、索引以及repetition的坐标
    return "{}:{}[{},{},{}]".format(atom.symbol, atom.index, offset[0], offset[1], offset[2])  #是不是返回index
    #return "{}:[{},{},{}]".format(atom.symbol, offset[0], offset[1], offset[2])
def reduce_node_symbol(atom,offset):
    return "{}".format(atom.index)


def add_atoms_node(graph, atoms, a1, o1, **kwargs):  #a1 就是index o1就是（x,y,z）
    #print("a1o1",a1,o1) #a1  索引，o1相应扩胞的量，1，1，0 应该是8个
    #print("nex",atoms[a1])  #获得原子信息-符号与位置
    #print("node_sym",node_symbol(atoms[a1],o1))
    graph.add_node(node_symbol(atoms[a1], o1), index=a1, central_ads=False, **kwargs)
def add_compare_atoms_node(graph, atoms, a1, o1, **kwargs):
    graph.add_node(reduce_node_symbol(atoms[a1], o1), symbol=atoms[a1].symbol, central_ads=False, **kwargs)

def add_atoms_edge(graph, atoms, a1, a2, o1, o2, adsorbate_atoms,count,**kwargs):
    #add_atoms_edge(full, atoms, index, neighbor, (x, y, z), (x + ox, y + oy, z + oz)，adsorbate_atoms) #是不是先不考虑吸附的情况呢？是如何判断吸附分子的index的？

    dist = 2 - (1 if a1 in adsorbate_atoms else 0) - (1 if a2 in adsorbate_atoms else 0)
    count.append([a1, a2])
    #index或者它的近邻在吸附原子的列表里  这个是用来赋予权重的？slab上的成键，是2，slab和吸附物的成键，是1，吸附物之间的成键，是0
    #dist = 2
    #return dist
    #print("----",atoms[a1], o1)
   # print("atoms[a1].index",atoms[a1].index)
    #print(node_symbol(atoms[a1],o1))
    #print(node_symbol(atoms[a2], o2))
    #print(bond_symbol(atoms,a1,a2))
    #print(dist)
    #print(atoms.get_distance(a1,a2))
    graph.add_edge(node_symbol(atoms[a1], o1),
                   node_symbol(atoms[a2], o2),
                   bond=bond_symbol(atoms, a1, a2),
                   index='{}:{}'.format(*sorted([a1, a2])),
                   #index='{}:{}'.format(*sorted([a1, a2])),
                   dist=dist,
                   dist_edge=atoms.get_distance(a1,a2,mic='True'),
                   ads_only=0 if (a1 in adsorbate_atoms and a2 in adsorbate_atoms) else 2,
                   **kwargs
                    )
    return count
def add_compare_atoms_edge(graph, atoms, a1, a2, o1, o2, adsorbate_atoms,count,**kwargs):
    #add_atoms_edge(full, atoms, index, neighbor, (x, y, z), (x + ox, y + oy, z + oz)，adsorbate_atoms) #是不是先不考虑吸附的情况呢？是如何判断吸附分子的index的？

    dist = 2 - (1 if a1 in adsorbate_atoms else 0) - (1 if a2 in adsorbate_atoms else 0)
    count.append([a1, a2])
    #index或者它的近邻在吸附原子的列表里  这个是用来赋予权重的？slab上的成键，是2，slab和吸附物的成键，是1，吸附物之间的成键，是0
    #dist = 2
    #return dist
    #print("----",atoms[a1], o1)
   # print("atoms[a1].index",atoms[a1].index)
    #print(node_symbol(atoms[a1],o1))
    #print(node_symbol(atoms[a2], o2))
    #print(bond_symbol(atoms,a1,a2))
    #print(dist)
    #print(atoms.get_distance(a1,a2))
    graph.add_edge(reduce_node_symbol(atoms[a1], o1),
                   reduce_node_symbol(atoms[a2], o2),
                   bond=bond_symbol(atoms, a1, a2),
                   index='{}:{}'.format(*sorted([a1, a2])),
                   #index='{}:{}'.format(*sorted([a1, a2])),
                   dist=dist,
                   #dist_edge=atoms.get_distance(a1,a2,mic='True'),
                   #ads_only=0 if (a1 in adsorbate_atoms and a2 in adsorbate_atoms) else 2,
                   **kwargs
                    )
    return count
def unique_chem_envs(chem_envs_groups, metadata=None, verbose=False):
    """Given a list of chemical environments, find the unique
    environments and keep track of metadata if required.

    This function exists largely to help with unique site detection
    but its performance will scale badly with extremely large numbers
    of chemical environments to check.  This can be split into parallel
    jobs.

    Args:
        chem_env_groups (list[list[networkx.Graph]]):
            Chemical environments to compare against each other
        metadata (list[object]):
            Corresponding metadata to keep with each chemical environment

    Returns:
        list[list[list[networkx.Graph]]]: A list of unique chemical environments
                                          with their duplicates
        list[list[object]]: A matching list of metadata
    """
    # Error checking, this should never really happen
    if len(chem_envs_groups) == 0:
        return [[], []]

    # We have metadata to manage
    if metadata is not None:
        if len(chem_envs_groups) != len(metadata):
            raise ValueError("Metadata must be the same length as\
                              the number of chem_envs_groups")

    # No metadata to keep track of
    if metadata is None:
        metadata = [None] * len(chem_envs_groups)

    # Keep track of known unique environments
    unique = []

    for index, env in enumerate(chem_envs_groups):
        for index2, (unique_indices, unique_env) in enumerate(unique):
            if verbose:
                print("Checking for uniqueness {:05d}/{:05d} {:05d}/{:05d}".format(index + 1, len(chem_envs_groups),
                                                                                   index2, len(unique)), end='\r')
            if compare_chem_envs(env, unique_env):
                unique_indices.append(index)
                break
        else:  # Was unique
            if verbose:
                print("")
            unique.append(([index], env))

    # Zip trick to split into two lists to return
    return zip(*[(env, [metadata[index] for index in indices]) for (indices, env) in unique])


def grid_iterator(grid):
    """Yield all of the coordinates in a 3D grid as tuples

    Args:
        grid (tuple[int] or int): The grid dimension(s) to
                                  iterate over (x or (x, y, z))

    Yields:
        tuple: (x, y, z) coordinates
    """
    if isinstance(grid, int): # Expand to 3D grid
        grid = (grid, grid, grid)

    for x in range(-grid[0], grid[0]+1):
        for y in range(-grid[1], grid[1]+1):
            for z in range(-grid[2], grid[2]+1):
                #print(x,y,z)
                yield (x, y, z)  #yield就是 return 返回一个值，并且记住这个返回的位置，下次迭代就从这个位置后(下一行)开始

def compare_chem_envs(chem_envs1, chem_envs2):
    """Compares two sets of chemical environments to see if they are the same
    in chemical identity.  Useful for detecting if two sets of adsorbates are
    in the same configurations.

    Args:
        chem_envs1 (list[networkx.Graph]): A list of chemical environments,
                                           including duplicate adsorbates
        chem_envs2 (list[networkx.Graph]): A list of chemical environments,
                                           including duplicate adsorbates

    Returns:
        bool: Is there a matching graph (site / adsorbate) for each graph?
    """
    # Do they have the same number of adsorbates?
    if len(chem_envs1) != len(chem_envs2):
        return False

    envs_copy = chem_envs2[:]  # Make copy of list

    # Check if chem_envs1 matches chem_envs2 by removing from envs_copy
    for env1 in chem_envs1:
        for env2 in envs_copy:
            if nx.is_isomorphic(env1, env2, edge_match=bond_match):
                # Remove this from envs_copy and move onto next env in chem_envs1
                envs_copy.remove(env2)
                break

    # Everything should have been removed from envs_copy if everything had a match
    if len(envs_copy) > 0:
        return False

    return True
def unique_adsorbates(chem_envs):
    """Removes duplicate adsorbates which occur when perodic
    boundary conditions detect the same adsorbate in two places. Each
    adsorbate graph has its atomic index checked to detect when PBC has
    created duplicates.

    Args:
        chem_env_groups (list[networkx.Graph]): Adsorbate/environment graphs

    Returns:
        list[networkx.Graph]: The unique adsorbate graphs
    """
    # Keep track of known unique adsorbate graphs
    unique = []
    for env in chem_envs:
        for unique_env in unique:
            if nx.is_isomorphic(env, unique_env, edge_match=bond_match, node_match=ads_match):
                break
        else: # Was unique
            unique.append(env)
    return unique

def atom_to_graph(atoms,neighbor_list,adsorbate_atoms,grid=(0,0,0),clean_graph=None,radius=0): #第一阶段，能够把Pd的对比出来就好了,grid 是指的扩胞的数量，grid的取值对计算速度有极大影响
    '''
    #总的而言，获得全图，并获得吸附物种的化学环境，grid是最耗时间的步骤
   :param atom: ase atom object
   :param grid:扩PBC的倍数，也就是超胞
   :param neighbor_list: 每个原子的成键近邻关系
   :param radius The radius for adsorbate graphs, this is a tunable parameter  0代表只和它接触的原子，需要考虑扩胞的因素,返回的原子包括自身
   :return:
   full graph of the whole system
   reduce graph, catkit graph for adstructure construted
   chem_env  local environment of the adsprbate
   compared graph: 相比于全图，少了周期性信息。
    '''
    distances = atoms.get_all_distances(mic=True)
    #print(distances)
    #print(atom.get_all_distances()[1])
    #print(type(atom.get_all_distances()[1])) #Return distances of all of the atoms with all of the atoms
    full=nx.Graph()  #创建一个没有节点和边的空图形,接下来就是把节点、距离、节点标识加到
    compare_full=nx.Graph()
    reduced_full=nx.Graph()
    #为全图添加节点，亦即原子
    for index,atom in enumerate(atoms):    #获得原子的索引和每个原子的位置信息
        for x,y,z in grid_iterator(grid):  #扩大边界后的原子位置信息
            add_atoms_node(full, atoms, index, (x, y, z))
            add_compare_atoms_node(compare_full, atoms, index, (x, y, z))
            reduced_full.add_node(index)
    #print(full.nodes)
    count = []
    for index, atom in enumerate(atoms):
        #print(index,atom)
        for x, y, z in grid_iterator(grid):
            neighbors, offsets = neighbor_list.get_neighbors(index)  # neighbors 返回的是和原子相连的原子序号，offset返回的是什么，另一个说法是displacement  #图判断不准确的问题就出在neighbor算法
            #print("indexneighboestotal",index,neighbors)
            for neighbor, offset in zip(neighbors, offsets):  #to calculate The positions of the neighbor atoms,description from ase
                #position=atoms[neighbor].position + np.dot(offset, atoms.get_cell())  #从这个看起来，offset起到的作用和周期性有关啊
                #print(index,atom,neighbor,offset,position)
                #print("before index neighbor", index, neighbor)
                ox, oy, oz = offset

                #print(ox+x,oy+y,oz+z)
                if not (-grid[0] <= ox + x <= grid[0]): #-1 1
                    continue             #如果 满足if 后面的条件，就跳过当前循环。物理意义，就在这个扩胞的边界里搜寻，如果是1，1，0，就是-1，0，+1 一共九个区域？
                if not (-grid[1] <= oy + y <= grid[1]):
                    continue
                if not (-grid[2] <= oz + z <= grid[2]):
                    continue

                #print(distances[index][neighbor])
                # This line ensures that only surface adsorbate bonds are accounted for that are less than 2.5 Å
                if distances[index][neighbor] > 2.5 and (bool(index in adsorbate_atoms) ^ bool(neighbor in adsorbate_atoms)):
                    #^符号的意义如果两个位中只有一位为 1，则将每个位设为 1。 bool是int的子类，如果是int，就返回true，注意0不是int
                    #这一行的意义是只考虑和界面成键的吸附物中的原子,问题在于ase的
                    continue
                    # print(index, neighbor, (x, y, z), (x + ox, y + oy, z + oz), adsorbate_atoms)  #adsprbateatom默认值是None，但实际上不能是None
                #print(index,neighbor)
                #print((atoms[index].index,atoms[neighbor].index))

                #print("deal index neighbor",index,neighbor)
                add_atoms_edge(full, atoms, index, neighbor, (x, y, z), (x + ox, y + oy, z + oz),adsorbate_atoms,count)
                add_compare_atoms_edge(compare_full, atoms, index, neighbor, (x, y, z), (x + ox, y + oy, z + oz),adsorbate_atoms,count)
                if atoms[index].index!=atoms[neighbor].index:
                    #print("000",(atoms[index].index, atoms[neighbor].index))
                    #print("index:",index,"neighbor:",neighbor,"atoms[index]:",atoms[index],"atoms[neighbor]:",atoms[neighbor])
                    reduced_full.add_edges_from([(atoms[index].index,atoms[neighbor].index)])
    #print(count)
    #print("len", len(count))

        #print(neighbor_list)
        #print(index,atom)
        #print(neighbor_list.get_neighbors(index))
    #print("beforecount",count)
    #为全图添加边，亦即连接关系
    # Add the case of surface-ads + ads-ads edges to clean graph case here
    if clean_graph:
        edges = [(u, v, d) for u, v,d in full.edges.data() if d["dist"] < 2]    ### Read all the edges, that are between adsorbate and surface (dist<2 condition)
        nodes = [(n, d) for n, d in full.nodes.data() if d["index"] in adsorbate_atoms]    ### take all the nodes that have an adsorbate atoms
        full=nx.Graph(clean_graph)
        full.add_nodes_from(nodes)
        full.add_edges_from(edges)

    # All adsorbates into single graph, no surface 创建只有吸附物的子图
    ads_nodes = None
    #print(adsorbate_atoms)
    ads_nodes = [node_symbol(atoms[index], (0, 0, 0)) for index in adsorbate_atoms]
    ads_graphs = nx.subgraph(full, ads_nodes)   #获得包含这些节点的子图，自然包含相应的连接关系了
    #print(ads_graphs.edges)

    ads_graphs = connected_component_subgraphs(ads_graphs)  #生成的是个generator，获得连通分量，连通分量：一个不完全连通的图中最大的连通部分？
    #print(list(ads_graphs))
    ##print(ads_graphs.nodes)
    #获得吸附物的化学环境，通常包含两层就够了，这个就是一个获取子图的操作。
    chem_envs = []
    for ads in ads_graphs:
        #print("ads",ads)
        initial = list(ads.nodes())[0]  #为什么要零呢？
        #print("initial",initial)
        full_ads = nx.ego_graph(full, initial, radius=1, distance="ads_only")  #ego_graph 返回在给定半径内以节点n为中心的邻域的诱导子图,distance使用指定的边缘数据键作为距离
        #print("full_ads.nodes",full_ads.nodes)

        new_ads = nx.ego_graph(full, initial, radius=(radius * 2) + 1, distance="dist")  #为什么radius要乘以2？
        #print("new_ads",new_ads.edges)
        new_ads = nx.Graph(nx.subgraph(full, list(new_ads.nodes())))  #放回节点为列表中那些的子图，没有这步行吗？没有连接信息？
        #print("new_ads", new_ads.edges)
        for node in ads.nodes():
            new_ads.add_node(node, central_ads=True)

        for node in full_ads.nodes():
            new_ads.add_node(node, ads=True)

        chem_envs.append(new_ads)

    chem_envs = unique_adsorbates(chem_envs)

    chem_envs.sort(key=lambda x: len(x.edges()))  #按键的长度sort
    #print(chem_envs)
    #print(type(chem_envs))
    #print(chem_envs[0].nodes)  #这个算法还有点问题，似乎无法准确的获取局部环境。

    #print(reduced_full.nodes)
    #print(reduced_full.edges)
    #print(full.nodes)
    #print(full.edges)
    #brances=nx.bfs_successors(reduced_full,0)  #从源返回广度优先搜索的后续项的迭代器,理解起来，就是从源头往外搜索一步，直至边缘节点，且搜索到第二层后，不会往回搜索
    #print(dict(brances))
    #print(full.nodes)
    #print("=============================")
    #print(full.edges)
    return full,compare_full,reduced_full,chem_envs



#atoms2=read(r'C:\Users\dell\Desktop\T\1.vasp')
#molecule=read(r'/share/home/zhengss/20210412niaosu/code_0816/test/CO2/1-CO2.vasp')


#test

#neighbor_list=get_neighbor_list(molecule)

##af=[]
##full1,compare,reduce_ull,chem=atom_to_graph(molecule,neighbor_list,adsorbate_atoms=[])
#print([node for node in full1.nodes])
#print([node for node in compare.nodes])