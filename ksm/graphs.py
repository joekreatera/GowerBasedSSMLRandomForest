import matplotlib.pyplot as plt
import numpy as np
import pprint
import pandas as pd
from numpy.random import default_rng
import math
import random

class GraphNodes():
    def __init__(self):
        self.nodes = {}
    
    def add_node(self, label, region):
        self.nodes[label] = region
    def get_node_region(self, label):
        return self.nodes[label]

class EdgeEndPoint():
    def __init__(self, label, value_min, value_max):
        self.label = label
        self.min_value = value_min
        self.max_value = value_max
    def __str__(self):
        return f'[{self.min_value}->{self.max_value}]'
        
class Edge():
    NO_RELATION = 0 # parent child relation
    PC_RELATION = 1 # parent child relation
    R_RELATION = 2 # relative relation 
    SELF_RELATION = 3 # to self relation
    def __init__(self, edgeEndPointA,edgeEndPointB, type): # for type use Edge.is_connection_valid
        self.connections = [edgeEndPointA,edgeEndPointB]
        self.type = type
    
    @staticmethod
    def is_connection_valid(edgeEndPointA,edgeEndPointB): 
        # print(edgeEndPointA, "vs", edgeEndPointB)
        if(edgeEndPointA == edgeEndPointB ):
            return Edge.SELF_RELATION
        # if one of them is with minValue = np.PINF or maxValue= np.NINF 
        if( edgeEndPointA.min_value == np.PINF or  edgeEndPointB.min_value == np.PINF):
            return Edge.NO_RELATION
        a_min_less_min_b =  edgeEndPointA.min_value <= edgeEndPointB.min_value
        a_min_great_min_b =  edgeEndPointA.min_value >= edgeEndPointB.min_value
        a_max_great_max_b =  edgeEndPointA.max_value >= edgeEndPointB.max_value
        a_max_less_max_b =  edgeEndPointA.max_value <= edgeEndPointB.max_value
        
        if a_min_less_min_b and a_max_great_max_b:
            return Edge.PC_RELATION
        
        if a_min_great_min_b and a_max_less_max_b:
            return Edge.PC_RELATION
        
        if a_min_great_min_b and a_max_great_max_b:
            return Edge.R_RELATION
        
        if a_min_less_min_b and a_max_less_max_b:
            return Edge.R_RELATION
        
        return Edge.NO_RELATION
        # any other thing should not be here

class EdgeSet():
    
    def __init__(self,graphNodes):
        self.graph = graphNodes
        self.edges = {}
        self.adjacency_by_node = {}
    
    @staticmethod
    def edge_relation_to_text(type):
        types = {
        Edge.PC_RELATION:'P',
        Edge.R_RELATION:'R',
        Edge.NO_RELATION:'N',
        Edge.SELF_RELATION:'S'
        }
        return types[type]
    
    def __str__(self):
        st = ""
        types = {
        Edge.PC_RELATION:'P',
        Edge.R_RELATION:'R',
        Edge.NO_RELATION:'N',
        Edge.SELF_RELATION:'S'
        }
        
        at_least_one = False
        for key in self.edges:
            at_least_one = True
            if( types[self.edges[key].type] != types[Edge.SELF_RELATION] ):
                st += f'{key} {types[self.edges[key].type]} {self.edges[key].connections[0]} <--> {self.edges[key].connections[1]} \n'
        
        if not at_least_one:
            st += f'({key} {types[self.edges[key].type]})'
        
        return st
    
    def get_parent_from_edge(self, labelA, labelB):
        """
        returns:
        0 if there is no parent child relation 
        1 if labelA is parent of labelB
        -1 is labelB is parent of labelA
        """
        edge = self.edges.get(f'{labelA}_{labelB}', None)
        if edge is None:
            edge = self.edges.get(f'{labelB}_{labelA}', None)
        if(edge is None):
            print(f"this should not happen as there should be an edge from {labelA} to {labelB}")
            return 0
        if edge.type in [Edge.R_RELATION , Edge.NO_RELATION , Edge.SELF_RELATION]:
            return 0
            
        edgeEndPointA = edge.connections[0]
        edgeEndPointB = edge.connections[1]
        
        a_min_less_min_b =  edgeEndPointA.min_value <= edgeEndPointB.min_value
        a_min_great_min_b =  edgeEndPointA.min_value >= edgeEndPointB.min_value
        a_max_great_max_b =  edgeEndPointA.max_value >= edgeEndPointB.max_value
        a_max_less_max_b =  edgeEndPointA.max_value <= edgeEndPointB.max_value
            
        if a_min_less_min_b and a_max_great_max_b:
            return 1 # is parent
            
        if a_min_great_min_b and a_max_less_max_b:
            return -1
        
        return 0
    
    def get_nodes_attached_to(self, node_label):
        node_list = self.adjacency_by_node.get(node_label,[]) 
        return node_list
        
        
    def add_edge(self, edgeEndPointA ,edgeEndPointB):
        
        t = Edge.is_connection_valid(edgeEndPointA ,edgeEndPointB)
        
        if t != Edge.NO_RELATION:
            edge = Edge(edgeEndPointA ,edgeEndPointB, t)
            self.edges[ f'{edgeEndPointA.label}_{edgeEndPointB.label}' ] = edge 
            
            node_list = self.adjacency_by_node.get(edgeEndPointA.label, []) 
            node_list.append(edgeEndPointB.label)
            self.adjacency_by_node[edgeEndPointA.label] = node_list
            
            node_list = self.adjacency_by_node.get(edgeEndPointB.label, []) 
            node_list.append(edgeEndPointA.label)
            self.adjacency_by_node[edgeEndPointB.label] = node_list
            
            return t
        return None
        