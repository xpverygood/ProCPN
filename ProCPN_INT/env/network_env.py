from collections import Counter
from statistics import pvariance
import gym
from sympy import N
import torch
import random
from copy import deepcopy
import numpy as np
import pdb, os, time, json
import networkx as nx
import matplotlib.pyplot as plt
from tianshou.data import Batch
from tianshou.env import DummyVectorEnv
import sys, math

random.seed(1)
class TelemetryEnv(gym.Env):
    def __init__(self, g = "topo/Nsfnet.graphml"):
        self.graphml_file = g
        self.G = nx.read_graphml(self.graphml_file)
        
        self.alpha = 0    
        self.beta= 1 
        self.norm_param=1e-10    
        self.norm_param_final=1e-1    
        self.bal_param = 1    
        self.telemetry_type = 5     
        self.fixed = 106   
        self.MTU = 1500    
        
        self.nodes = self.G.nodes()  
        self.nodes_num = len(self.nodes)
        
        self.edge_index_0 = [edge for edge in self.G.edges()]
        self.edge_index_0 = [(int(u), int(v)) for u, v in self.edge_index_0]
        self.edge_index = torch.LongTensor(np.array(self.edge_index_0).T) 
        self.edge_len = len(self.edge_index_0)
        
        self.node_port = {} 
        self.edge_port_index = []   
        self.get_node_port()
        self.get_edge_port_index()

        self.port_req = [] 
        self.nodes_feature = np.zeros((self.nodes_num, self.telemetry_type))  
        self.port_edge_index = np.zeros(self.edge_len*2)
        self.get_port_req()
        self.get_nodes_feature()
        self.get_port_edge()

        self.C_m = []
        self.get_C_m()

        self._observation_space = gym.Space(shape=list(self.nodes_feature.shape))
        # print("obv_space size: {}".format(self.observation_space.shape))

        self._action_space = gym.spaces.Discrete(self.edge_len*2+1) #port number
        # print("act_space size: {}".format(self.action_space.n))
        
        self.mask = None
        
        self._num_steps = 0
        self._terminated = False
        self._laststate = dict(nodes_feature = deepcopy(self.nodes_feature), probe = [[]]) 
        self._lastobs = deepcopy(self.nodes_feature)
        self._remain_edge = deepcopy(self.edge_port_index)
        # Define the number of steps per episode
        self._steps_per_episode = self.edge_len*2
        
        
    @property
    def observation_space(self):
        # Return the observation space
        return self._observation_space

    @property
    def action_space(self):
        # Return the action space
        return self._action_space

    def step(self, action):
        # Check if episode has ended
        assert not self._terminated, "One episodic has terminated"
        # Calculate reward based on last state and action taken
        if action == 0:
            self._laststate["probe"].append([]) 
            
            mask_list = np.zeros(self.edge_len*2+1, dtype=int)
            mask_list[np.array(self._remain_edge).flatten()+1] = 1
            
            mask_list[0] = 0   
            self.mask = mask_list
            
        else:
            action -= 1 
            action_edge_index = int(self.port_edge_index[action])    
            remain_edge_index = [i for i, l in enumerate(self._remain_edge) if action in l][0]  
            action_edge = self.edge_port_index[action_edge_index] 
            self._laststate["probe"][-1].append(int(action)) 
            x_1 = [x for x in action_edge if x != action][0]    
            self._laststate["probe"][-1].append(x_1)
            
            port_req_sat = self.port_req[action_edge_index] 
            for x in port_req_sat:
                self._lastobs[self.edge_index[0, action_edge_index], x-1] -=1
                self._lastobs[self.edge_index[1, action_edge_index], x-1] -=1
                
            self._remain_edge = np.delete(self._remain_edge, remain_edge_index, axis=0)
            self.clc_mask(x_1)
         
        reward = self.clc_rwd_cost()

        self._num_steps += 1
        # Check if episode should end based on number of steps taken
        if self._num_steps >= self._steps_per_episode or np.all(self._lastobs == 0):  
            self._terminated = True
        # Information about number of steps taken
        info = {'num_steps': self._num_steps, 'mask': self.mask}
        
        if self._terminated:
            reward += self.clc_final_rwd()
        
        return self._lastobs, reward, self._terminated, info

    def reset(self):
        # Reset the environment to its initial state
        self._num_steps = 0
        self._terminated = False
        state = self.nodes_feature
        self._laststate = dict(nodes_feature = deepcopy(self.nodes_feature), probe = [[]])
        self._lastobs = deepcopy(self.nodes_feature)
        self._remain_edge = deepcopy(self.edge_port_index)
        
        mask_list = [1]*(self.edge_len*2+1)
        mask_list[0] = 0
        self.mask = mask_list
        
        return state, {'num_steps': self._num_steps, 'mask': self.mask}
        
    
    def clc_rwd_cost(self):   
        cu_probe = self._laststate["probe"][-1] 
        cu_probe = cu_probe[0:-1]  
        cost = 0
        for x in cu_probe:
            cu_edge_index = int(self.port_edge_index[x])
            cu_prot_item = self.port_req[cu_edge_index]
            for y in cu_prot_item:
                cost += self.C_m[y-1]
        rwd = -cost * self.norm_param
        
        return rwd
    
    def clc_final_rwd(self):
        
        probe_len = [len(x) for x in self._laststate["probe"]]  
        max_probe_len = max(probe_len)  
        probe_num = len(probe_len)  
        probe_var = [x for x in probe_len if x != 0]   
        if probe_num == 1:  
            probe_var = 0
        else:
            probe_var = pvariance(probe_len)
        rwd = -(self.alpha*probe_var + self.beta*probe_num) * self.norm_param_final + self.bal_param
        
        return rwd
    
    def clc_mask(self, x):
        
        mask_list = [0]*(self.edge_len*2+1)    #0~infeasible,1~feasible

        cu_probe = self._laststate["probe"][-1] 
        occupancy = 0
        for x in cu_probe:
            cu_edge_index = int(self.port_edge_index[x])
            cu_prot_item = self.port_req[cu_edge_index]
            for y in cu_prot_item:
                occupancy += self.C_m[y-1]
        occupancy += self.fixed

        if occupancy > self.MTU - 2*sum(self.C_m):  
            mask_list[0] = 1    
        else:
            other_ports = None
            for k, v in self.node_port.items():    
                if x in v:
                    other_ports = deepcopy(v)
                    other_ports.remove(x)
                    break
            
            for p in other_ports:       
                if any(p in _ for _ in self._remain_edge):
                    mask_list[p+1] = 1
    
            mask_list[0] = 1

        self.mask = mask_list
    
    def get_port_req(self):
        self.port_req = []
        edge_num = self.edge_index.shape[1] 
        for _ in range(edge_num):
            items_num = random.randint(2, self.telemetry_type) 
            edgex_item =random.sample(range(1, self.telemetry_type+1), items_num) 
            self.port_req.append(edgex_item)
        
    def get_nodes_feature(self):    
        self.nodes_feature = np.zeros((self.nodes_num, self.telemetry_type))
        edge_num = self.edge_index.shape[1]
        for edge in range(edge_num):    
            for x in self.port_req[edge]:
                self.nodes_feature[self.edge_index[0, edge], x-1] +=1
                self.nodes_feature[self.edge_index[1, edge], x-1] +=1
        
    def get_node_port(self):
        flat_list = [item for sublist in self.edge_index_0 for item in sublist]
        element_counts = dict(Counter(flat_list))

        for i in range(self.nodes_num-1):
            element_counts[i+1] += element_counts[i]

        for i in range(self.nodes_num):
            if i != self.nodes_num - 1:
                element_counts[self.nodes_num - i -1] = list(range(element_counts[self.nodes_num - i -2], element_counts[self.nodes_num - i -1]))
            else:
                element_counts[self.nodes_num - i -1] = list(range(element_counts[self.nodes_num - i -1]))
        
        self.node_port = element_counts
        
    def get_edge_port_index(self):
        node_port_0 = deepcopy(self.node_port)
        edge_index_00 = deepcopy(self.edge_index_0)
        edge_index_00 = np.array(edge_index_00)
        for edge in edge_index_00:
            x1 = node_port_0[edge[0]].pop(0)
            edge[0] = x1

            x2 = node_port_0[edge[1]].pop(0)
            edge[1] = x2

        self.edge_port_index = edge_index_00
        
    def get_C_m(self):
        for i in range(self.telemetry_type):
            self.C_m = [1, 2, 3, 4, 6]

    def get_port_edge(self):
        i = 0
        for x in self.edge_port_index:
            self.port_edge_index[x[0]] = i
            self.port_edge_index[x[1]] = i
            i += 1

    def overheads(self, port_req, probe):
        overheads = 0
        for probex in probe:
            overheads_p = 0
            for idx, port in enumerate(probex):
                overheads_m = 0
                probex_len = len(probex)
                edge_idx = int(self.port_edge_index[port])
                items = port_req[edge_idx]
                for item in items:
                    overheads_m += self.C_m[item-1]
                overheads_p += overheads_m * math.ceil((probex_len-idx-1)/2)
            overheads += overheads_p
        return overheads

    def handle_probe(self, probe):  
        probe_dp = []
        for probex in probe:
            probe_dp.append(probex[0::2]+[probex[-1]])
        return probe_dp

    def handle_sw(self, sw):    
        def find_edge_index(start, end, edge_dict):
            for index, p in enumerate(edge_dict):
                order_ = 0
                if (p[0] == start and p[1] == end) or (p[0] == end and p[1] == start):
                    if (p[0] == start and p[1] == end):
                        order_ = 1 
                    if (p[0] == end and p[1] == start):
                        order_ = 0 
                    return index, order_
            return None
        
        all_port_lists = []

        for point_sequence in sw:
            edge_indices = []
            order_list = []
            for i in range(len(point_sequence) - 1):
                start_point = point_sequence[i]
                end_point = point_sequence[i + 1]
                edge_index, order_ = find_edge_index(start_point, end_point, self.edge_index_0)
                if edge_index is not None:
                    edge_indices.append(edge_index)
                    order_list.append(order_)

            port_list = []
            for idx, edge_index in enumerate(edge_indices):
                ports = self.edge_port_index[edge_index]
                if order_list[idx] == 1:
                    port_list.extend(ports)
                else:
                    port_list.append(ports[1])
                    port_list.append(ports[0])
            
            all_port_lists.append(port_list)
        
        return all_port_lists

    def seed(self, seed = None):
        random.seed(seed)

def make_telemetry_env(training_num = 0, test_num = 0):
    
    env = TelemetryEnv()
    env.seed(0)
    
    train_envs, test_envs = None, None
    if training_num:
        # Create multiple instances of the environment for training
        train_envs = DummyVectorEnv(
            [lambda: TelemetryEnv() for _ in range(training_num)])
        train_envs.seed(0)

    if test_num:
        # Create multiple instances of the environment for testing
        test_envs = DummyVectorEnv(
            [lambda: TelemetryEnv() for _ in range(test_num)])
        test_envs.seed(0)
        
    return env, train_envs, test_envs

def probe_data(probe):
    probe_len = [len(x)/2 for x in probe]  
    max_probe_len = max(probe_len) 
    probe_num = len(probe_len)  
    probe_var = pvariance(probe_len)

    return probe_num, max_probe_len, probe_len, sum(probe_len),probe_var

