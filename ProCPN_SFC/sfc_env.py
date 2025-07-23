
import gym
from gym import spaces
import networkx as nx
import numpy as np
import json
import random
from copy import deepcopy, copy
from collections import defaultdict
import os
import math
import itertools # Needed for islice in k-shortest paths
from typing import Any, Dict, List, Optional, Tuple, Union # For type hinting

from tianshou.env import DummyVectorEnv

VNF_TYPES = [f"VNF{i+1}" for i in range(10)]
DEFAULT_VNF_CPU_CONSUMPTION = {vnf: random.randint(1, 10) for vnf in VNF_TYPES}
MAX_RANDOM_BANDWIDTH = 80.0
DEFAULT_NODE_CPU = 40.0
DEFAULT_INSTANCE_CAPACITY = 5
INVALID_ACTION_PENALTY = -20.0
DEFAULT_K_SHORTEST = 5
REWARD_VALID_PLACEMENT = 0.1; REWARD_REUSE_BONUS = 0.2; REWARD_SFC_ROUTING_SUCCESS = 20

class SFCEnv(gym.Env):
    metadata = {"render_modes": [], "render_fps": 4}

    def __init__(
        self,
        sfc_requests_file: str,
        topology_file: str,
        vnf_cpu_consumption: dict = None,
        vnf_instance_capacity: int = DEFAULT_INSTANCE_CAPACITY,
        max_allowed_instances: int = 50,
        invalid_action_penalty: float = INVALID_ACTION_PENALTY,
        reward_weights: dict = {'accept': 5.0, 'deploy': 0.25, 'hops': 0.05},
        default_node_cpu: float = DEFAULT_NODE_CPU,
        decision_schedule_type: str = 'sfc_polling',
        k_shortest_paths: int = DEFAULT_K_SHORTEST,
        initial_link_bandwidths: Optional[Dict[tuple, float]] = None
    ):
        super().__init__()
        self.sfc_requests_file = sfc_requests_file; self.topology_file = topology_file; self.initial_node_cpu_capacity = default_node_cpu
        self.vnf_cpu_consumption = vnf_cpu_consumption or DEFAULT_VNF_CPU_CONSUMPTION; self.max_allowed_instances = max_allowed_instances
        self.invalid_action_penalty = invalid_action_penalty; self.reward_weights = reward_weights
        self.default_node_cpu = default_node_cpu; self.vnf_instance_capacity = vnf_instance_capacity
        self.decision_schedule_type = decision_schedule_type; self.k_shortest = k_shortest_paths
        self._load_sfc_requests(); self._load_topology(initial_link_bandwidths)
        all_vnf_req_types = set(vnf for req in self.sfc_requests for vnf in req['vnf_seq']); known_vnf_types = list(self.vnf_cpu_consumption.keys()); unknown_types = all_vnf_req_types - set(known_vnf_types)
        if unknown_types: print(f"Warning: Unknown VNF types: {unknown_types}. Assigning default CPU=1."); [self.vnf_cpu_consumption.setdefault(utype, 1.0) for utype in unknown_types]
        self.vnf_type_list = sorted(list(self.vnf_cpu_consumption.keys())); self.vnf_type_to_id = {vnf_type: i for i, vnf_type in enumerate(self.vnf_type_list)}; self.num_vnf_types = len(self.vnf_type_list)
        self.action_space = spaces.Discrete(self.num_nodes)
        obs_size = (self.num_nodes + self.num_edges + self.max_allowed_instances * 4 + 6)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float32)
        self.current_node_cpu = None; self.current_link_bandwidth = None; self.vnf_instances = None; self.sfc_states = None; self.decision_queue = None; self.current_decision_ptr = None; self.next_instance_id = None; self.num_successful_sfcs = None; self.total_vnf_deployments = None; self.total_routing_hops = None

    def _load_sfc_requests(self):
        with open(self.sfc_requests_file, 'r') as f: self.sfc_requests = json.load(f); self.num_sfcs = len(self.sfc_requests)
        for req in self.sfc_requests:
             try: req['source_node'] = int(req['source_node']); req['destination_node'] = int(req['destination_node'])
             except ValueError: print(f"Error converting nodes for SFC {req.get('sfc_id')}"); raise

    def _load_topology(self, initial_link_bandwidths: Optional[Dict[tuple, float]] = None):
        try:
            self.G = nx.read_graphml(self.topology_file, node_type=int)
            if isinstance(list(self.G.nodes())[0], str): self.G = nx.relabel_nodes(self.G, {n: int(n) for n in self.G.nodes()})
            self.nodes = sorted(list(self.G.nodes())); self.num_nodes = len(self.nodes); self.edges = sorted([tuple(sorted((int(u), int(v)))) for u, v in self.G.edges()]); self.num_edges = len(self.edges)
            self.initial_node_resources = {}; self._max_node_cpu = 0.0; self._max_link_bw = 0.0
            for node in self.nodes: cpu = float(self.G.nodes[node].get('cpu', self.default_node_cpu)); self.initial_node_resources[node] = cpu; self._max_node_cpu = max(self._max_node_cpu, cpu)
            if initial_link_bandwidths is not None:
                self.initial_link_resources = deepcopy(initial_link_bandwidths)
                if self.initial_link_resources: self._max_link_bw = max(self.initial_link_resources.values())
            else:
                self.initial_link_resources = {}
                for u_orig, v_orig in self.G.edges(): u, v = tuple(sorted((int(u_orig), int(v_orig)))); bw = float(random.randint(40, int(MAX_RANDOM_BANDWIDTH))); self.initial_link_resources[(u, v)] = bw
                if self.initial_link_resources: self._max_link_bw = max(self.initial_link_resources.values())
            if self._max_node_cpu <= 0: self._max_node_cpu = 1.0
            if self._max_link_bw <= 0: self._max_link_bw = 1.0
        except Exception as e: print(f"Error loading topology: {e}"); raise

    def _create_decision_queue(self):
        queue = []; schedule_type = self.decision_schedule_type
        if schedule_type == 'vnf_polling':
            max_vnfs = 0
            if self.sfc_requests: max_vnfs = max((len(sfc['vnf_seq']) for sfc in self.sfc_requests if sfc.get('vnf_seq')), default=0)
            for vnf_idx in range(max_vnfs):
                for sfc_list_idx in range(self.num_sfcs):
                    if vnf_idx < len(self.sfc_requests[sfc_list_idx].get('vnf_seq',[])): queue.append((sfc_list_idx, vnf_idx))
        elif schedule_type == 'sfc_polling':
            for sfc_list_idx in range(self.num_sfcs):
                for vnf_idx_in_sfc in range(len(self.sfc_requests[sfc_list_idx].get('vnf_seq',[]))): queue.append((sfc_list_idx, vnf_idx_in_sfc))
        else: raise ValueError(f"Unknown decision schedule type: {schedule_type}")
        self.decision_queue = queue
        self.chromosome_length = len(self.decision_queue)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed);
        if seed is not None: self.seed(seed); np.random.seed(seed); random.seed(seed)
        self.current_node_cpu = deepcopy(self.initial_node_resources); self.current_link_bandwidth = deepcopy(self.initial_link_resources)
        self.vnf_instances = {}; self.next_instance_id = 0; self.sfc_states = []
        for i, req in enumerate(self.sfc_requests): self.sfc_states.append({'id': req['sfc_id'], 'status': 'pending', 'vnf_nodes': [req['source_node']], 'current_vnf_idx': 0, 'total_hops': 0, 'routing_path': [], 'fail_reason': '' })
        self._create_decision_queue(); self.current_decision_ptr = 0; self.num_successful_sfcs = 0; self.total_vnf_deployments = 0; self.total_routing_hops = 0
        observation = self._build_observation(); info = {}
        return observation, info

    def step(self, action):
        terminated = False; truncated = False; reward = 0.0; info = {}
        current_sfc_idx, current_vnf_idx_in_sfc = self._get_current_decision_focus()
        if current_sfc_idx is None: terminated = True; reward = self._calculate_final_reward(); observation = self._build_observation(None, None); info['final_status'] = deepcopy(self.sfc_states); return observation, reward, terminated, truncated, info
        sfc_state = self.sfc_states[current_sfc_idx]; sfc_req = self.sfc_requests[current_sfc_idx]
        if not sfc_state['status'].startswith('pending') and not sfc_state['status'].startswith('placing'): observation = self._build_observation(); return observation, 0.0, False, False, info
        if sfc_state['status'] == 'pending': sfc_state['status'] = 'placing'
        if current_vnf_idx_in_sfc >= len(sfc_req['vnf_seq']): reward = self.invalid_action_penalty; terminated = True; sfc_state['status'] = 'failed_logic'; sfc_state['fail_reason'] = f'VNF index {current_vnf_idx_in_sfc} out of bounds'; info['error'] = sfc_state['fail_reason']; info['invalid_action'] = True; observation = self._build_observation(current_sfc_idx, current_vnf_idx_in_sfc); return observation, reward, terminated, truncated, info
        vnf_type = sfc_req['vnf_seq'][current_vnf_idx_in_sfc]; vnf_cpu_req = self.vnf_cpu_consumption.get(vnf_type, 1.0); previous_node = sfc_state['vnf_nodes'][-1]
        if not (0 <= action < self.num_nodes): reward = self.invalid_action_penalty; terminated = True; sfc_state['status'] = 'failed_placement'; sfc_state['fail_reason'] = f"Invalid action index {action}"; info['error'] = sfc_state['fail_reason']; info['invalid_action'] = True; observation = self._build_observation(current_sfc_idx, current_vnf_idx_in_sfc); return observation, reward, terminated, truncated, info
        target_node = self.nodes[action]; potential_reuse_instance_id = -1; can_reuse = False
        for inst_id, inst_info in self.vnf_instances.items():
            if inst_info['node'] == target_node and inst_info['type'] == vnf_type:
                 if inst_info['load'] < inst_info['capacity']: can_reuse = True; potential_reuse_instance_id = inst_id; break
        placement_successful = False; is_invalid_placement = False; is_invalid_reason = ""; action_type = ""
        path_segment_exists = True
        try:
            if previous_node != target_node: _ = nx.shortest_path(self.G, source=previous_node, target=target_node)
        except (nx.NetworkXNoPath, nx.NodeNotFound): is_invalid_placement = True; path_segment_exists = False; is_invalid_reason = f"No path segment {previous_node}->{target_node}"
        if not is_invalid_placement and path_segment_exists:
            if can_reuse:
                if self.vnf_instances[potential_reuse_instance_id]['load'] >= self.vnf_instances[potential_reuse_instance_id]['capacity']: is_invalid_placement = True; is_invalid_reason = f"Instance {potential_reuse_instance_id} at full capacity"
                else: placement_successful = True; action_type = 'IMPLICIT_REUSE'
            else:
                if self.current_node_cpu.get(target_node, 0) < vnf_cpu_req: is_invalid_placement = True; is_invalid_reason = f"Insufficient CPU at node {target_node}"
                elif self.next_instance_id >= self.max_allowed_instances: is_invalid_placement = True; is_invalid_reason = f"Reached max allowed instances"
                else: placement_successful = True; action_type = 'DEPLOY'
        if is_invalid_placement:
            reward = self.invalid_action_penalty; terminated = True; sfc_state['status'] = 'failed_placement'; sfc_state['fail_reason'] = is_invalid_reason; info['error'] = is_invalid_reason; info['invalid_action'] = True; observation = self._build_observation(current_sfc_idx, current_vnf_idx_in_sfc)
        else:
            intermediate_reward = REWARD_VALID_PLACEMENT;
            if action_type == 'IMPLICIT_REUSE': intermediate_reward += REWARD_REUSE_BONUS
            reward = intermediate_reward; info['invalid_action'] = False; self._update_resources(action_type, target_node, potential_reuse_instance_id, vnf_cpu_req, current_sfc_idx, current_vnf_idx_in_sfc)
            sfc_state['vnf_nodes'].append(target_node); sfc_state['current_vnf_idx'] += 1
            if sfc_state['current_vnf_idx'] == len(sfc_req['vnf_seq']):
                sfc_state['status'] = 'routing'
                routing_successful, route_hops, full_route_nodes, routing_fail_reason = self._route_sfc_k_shortest(current_sfc_idx, k=self.k_shortest)
                if routing_successful:
                    reward += REWARD_SFC_ROUTING_SUCCESS; sfc_state['status'] = 'success';
                    sfc_state['total_hops'] = route_hops # Store correct hops
                    sfc_state['routing_path'] = full_route_nodes # Store correct path
                    self.num_successful_sfcs += 1; self.total_routing_hops += route_hops
                else: sfc_state['status'] = 'failed_routing'; sfc_state['fail_reason'] = routing_fail_reason
            self.current_decision_ptr += 1
            next_sfc_idx_for_obs, next_vnf_idx_for_obs = self._get_next_valid_decision()
            if self.current_decision_ptr >= len(self.decision_queue): terminated = True; reward += self._calculate_final_reward(); info['final_status'] = deepcopy(self.sfc_states); next_sfc_idx_for_obs, next_vnf_idx_for_obs = None, None
            observation = self._build_observation(next_sfc_idx_for_obs, next_vnf_idx_for_obs)
        return observation, reward, terminated, truncated, info

    def _get_next_valid_decision(self):
        while self.current_decision_ptr < len(self.decision_queue):
            sfc_idx, vnf_idx = self.decision_queue[self.current_decision_ptr];
            if 0 <= sfc_idx < len(self.sfc_states) and (self.sfc_states[sfc_idx]['status'] == 'pending' or self.sfc_states[sfc_idx]['status'] == 'placing'): return sfc_idx, vnf_idx
            self.current_decision_ptr += 1
        return None, None

    def _get_current_decision_focus(self):
        if self.current_decision_ptr < len(self.decision_queue):
             sfc_idx, vnf_idx = self.decision_queue[self.current_decision_ptr];
             if 0 <= sfc_idx < len(self.sfc_states) and (self.sfc_states[sfc_idx]['status'] == 'pending' or self.sfc_states[sfc_idx]['status'] == 'placing'): return sfc_idx, vnf_idx
             else: return self._get_next_valid_decision()
        return None, None

    def _build_observation(self, current_sfc_idx=None, current_vnf_idx_in_sfc=None):
        if current_sfc_idx is None and current_vnf_idx_in_sfc is None: current_sfc_idx, current_vnf_idx_in_sfc = self._get_current_decision_focus()
        obs_list = []; scaled_node_cpu = [self.current_node_cpu.get(node, 0) / self._max_node_cpu if self._max_node_cpu > 0 else self.current_node_cpu.get(node, 0) for node in self.nodes]; scaled_link_bw = [self.current_link_bandwidth.get(edge, 0) / self._max_link_bw if self._max_link_bw > 0 else self.current_link_bandwidth.get(edge, 0) for edge in self.edges]; obs_list.extend(scaled_node_cpu); obs_list.extend(scaled_link_bw)
        instance_features = []; instance_count = 0; sorted_instance_ids = sorted(self.vnf_instances.keys())
        for inst_id in sorted_instance_ids:
             if instance_count < self.max_allowed_instances: inst = self.vnf_instances[inst_id]; capacity = inst['capacity']; load = inst['load']; rem_cap_scaled = (capacity - load) / capacity if capacity > 0 else 0.0; node_id_scaled = float(inst['node']) / self.num_nodes if self.num_nodes > 0 else float(inst['node']); instance_features.extend([float(inst_id) / self.max_allowed_instances if self.max_allowed_instances > 0 else float(inst_id), float(inst['type_id']), node_id_scaled, rem_cap_scaled]); instance_count += 1
             else: break
        padding_per_instance = 4; instance_features.extend([0.0] * padding_per_instance * (self.max_allowed_instances - instance_count)); obs_list.extend(instance_features)
        focus_features = [0.0] * 6
        if current_sfc_idx is not None and 0 <= current_sfc_idx < len(self.sfc_requests):
            sfc_req = self.sfc_requests[current_sfc_idx]; sfc_state = self.sfc_states[current_sfc_idx]
            if (sfc_state['status'] == 'placing' or sfc_state['status'] == 'pending') and current_vnf_idx_in_sfc is not None and current_vnf_idx_in_sfc < len(sfc_req['vnf_seq']):
                 vnf_type = sfc_req['vnf_seq'][current_vnf_idx_in_sfc]; vnf_type_id = self.vnf_type_to_id.get(vnf_type, -1); vnf_cpu_req = self.vnf_cpu_consumption.get(vnf_type, 0); sfc_bw_req = sfc_req['bandwidth']; prev_node = sfc_state['vnf_nodes'][-1]
                 cpu_req_scaled = vnf_cpu_req / self._max_node_cpu if self._max_node_cpu > 0 else vnf_cpu_req; bw_req_scaled = sfc_bw_req / self._max_link_bw if self._max_link_bw > 0 else sfc_bw_req; prev_node_scaled = float(prev_node) / self.num_nodes if self.num_nodes > 0 else float(prev_node)
                 focus_features = [ float(current_sfc_idx) / self.num_sfcs if self.num_sfcs > 0 else float(current_sfc_idx), float(current_vnf_idx_in_sfc) / len(sfc_req['vnf_seq']) if len(sfc_req['vnf_seq']) > 0 else float(current_vnf_idx_in_sfc), float(vnf_type_id), cpu_req_scaled, bw_req_scaled, prev_node_scaled ]
        obs_list.extend(focus_features); final_obs = np.array(obs_list, dtype=np.float32)
        if np.isnan(final_obs).any() or np.isinf(final_obs).any(): print(f"!!! ERROR: NaN/Inf generated during observation building/scaling at focus ({current_sfc_idx},{current_vnf_idx_in_sfc})!!!"); final_obs = np.nan_to_num(final_obs)
        return final_obs

    def _route_sfc_k_shortest(self, sfc_list_idx, k=5):
        sfc_state = self.sfc_states[sfc_list_idx]; sfc_req = self.sfc_requests[sfc_list_idx];
        placement_sequence = sfc_state['vnf_nodes'] + [sfc_req['destination_node']]; bw_req = sfc_req['bandwidth']
        total_hops = 0; links_to_commit = set(); full_route_nodes = [placement_sequence[0]] # Start path
        last_fail_reason = "Unknown routing failure"

        try:
            for i in range(len(placement_sequence) - 1):
                u = placement_sequence[i]; v = placement_sequence[i+1];
                if u == v: continue
                found_valid_segment_path = False; best_segment_path_nodes = None
                k_shortest_paths_generator = nx.shortest_simple_paths(self.G, source=u, target=v)
                path_checked_count = 0
                for segment_path_candidate in itertools.islice(k_shortest_paths_generator, k):
                    path_checked_count += 1; segment_hops = len(segment_path_candidate) - 1; segment_links_ok = True; current_segment_links = set()
                    if segment_hops < 0: segment_hops = 0
                    for j in range(segment_hops):
                        link_u, link_v = tuple(sorted((segment_path_candidate[j], segment_path_candidate[j+1]))); link_key = (link_u, link_v)
                        if self.current_link_bandwidth.get(link_key, 0) < bw_req: segment_links_ok = False; break
                        current_segment_links.add(link_key)
                    if segment_links_ok: best_segment_path_nodes = segment_path_candidate; total_hops += segment_hops; links_to_commit.update(current_segment_links); found_valid_segment_path = True; break
                if not found_valid_segment_path:
                    last_fail_reason = f"No path with sufficient BW found among {k} shortest ({path_checked_count} checked) for segment {u}->{v}"
                    return False, 0, [], last_fail_reason # Return 4 values
                if best_segment_path_nodes: full_route_nodes.extend(best_segment_path_nodes[1:])
                else: return False, 0, [], f"Internal logic error: path reported valid but not stored for {u}->{v}" # Return 4 values

            for link_key in links_to_commit:
                if link_key in self.current_link_bandwidth: self.current_link_bandwidth[link_key] -= bw_req; self.current_link_bandwidth[link_key] = max(0, self.current_link_bandwidth[link_key])

            return True, total_hops, full_route_nodes, "" # Return 4 values on success

        except nx.NetworkXNoPath: return False, 0, [], f"No path exists at all for segment {u}->{v}" # Return 4 values
        except Exception as e: print(f"Error during k-shortest routing for SFC {sfc_req['sfc_id']}: {e}"); return False, 0, [], f"Routing algorithm error: {e}" # Return 4 values

    def _get_route(self, source, target):
        if source == target: return [source];
        try: path = nx.shortest_path(self.G, source=source, target=target); return path
        except nx.NetworkXNoPath: return None
        except nx.NodeNotFound: print(f"Warning: Node not found during routing: {source} or {target}"); return None

    def _check_path_bandwidth(self, path_nodes, bw_req):
        if len(path_nodes) <= 1: return True
        for i in range(len(path_nodes) - 1): u, v = tuple(sorted((path_nodes[i], path_nodes[i+1])));
        if self.current_link_bandwidth.get((u, v), 0) < bw_req: return False
        return True

    def _commit_path_bandwidth(self, path_nodes, bw_req):
         if len(path_nodes) <= 1: return
         for i in range(len(path_nodes) - 1): u, v = tuple(sorted((path_nodes[i], path_nodes[i+1]))); link_key = (u, v)
         if link_key in self.current_link_bandwidth: self.current_link_bandwidth[link_key] -= bw_req; self.current_link_bandwidth[link_key] = max(0, self.current_link_bandwidth[link_key])

    def _check_resources(self, action_type, target_node, target_instance_id, vnf_cpu_req):
        if action_type == 'DEPLOY':
            if self.current_node_cpu.get(target_node, 0) < vnf_cpu_req: return False
            if self.next_instance_id >= self.max_allowed_instances: return False
        elif action_type == 'IMPLICIT_REUSE':
             if target_instance_id not in self.vnf_instances: return False
             instance_info = self.vnf_instances[target_instance_id]
             if instance_info['load'] >= instance_info['capacity']: return False
        else: return False
        return True

    def _update_resources(self, action_type, target_node, target_instance_id, vnf_cpu_req, sfc_list_idx, vnf_idx_in_sfc):
        if action_type == 'DEPLOY':
            self.current_node_cpu[target_node] -= vnf_cpu_req; self.current_node_cpu[target_node] = max(0, self.current_node_cpu[target_node])
            instance_id = self.next_instance_id
            if sfc_list_idx is not None and 0 <= sfc_list_idx < len(self.sfc_requests) and \
               vnf_idx_in_sfc is not None and 0 <= vnf_idx_in_sfc < len(self.sfc_requests[sfc_list_idx]['vnf_seq']): vnf_type = self.sfc_requests[sfc_list_idx]['vnf_seq'][vnf_idx_in_sfc]
            else: vnf_type = "Unknown"
            self.vnf_instances[instance_id] = {'type': vnf_type, 'type_id': self.vnf_type_to_id.get(vnf_type, -1), 'node': target_node, 'capacity': self.vnf_instance_capacity, 'load': 1}
            self.next_instance_id += 1; self.total_vnf_deployments += 1
        elif action_type == 'IMPLICIT_REUSE':
            if target_instance_id in self.vnf_instances: self.vnf_instances[target_instance_id]['load'] += 1

    def _calculate_final_reward(self):
        num_successful_sfcs = self.num_successful_sfcs; deploy_count = self.total_vnf_deployments; total_hops = self.total_routing_hops
        R_accept = self.reward_weights.get('accept', 1.0) * num_successful_sfcs; C_deploy = self.reward_weights.get('deploy', 0.05) * deploy_count; C_hops = self.reward_weights.get('hops', 0.01) * total_hops
        final_reward = R_accept - C_deploy - C_hops
        return final_reward

    def render(self, mode="human"): pass
    def close(self): pass
    def seed(self, seed=None):
        if seed is not None: random.seed(seed); np.random.seed(seed)
        return [seed]

def make_sfc_env(sfc_requests_file, topology_file, training_num=1, test_num=1, seed=0, **kwargs):
    # print("Generating initial random bandwidths for all environments...")
    init_env_kwargs = kwargs.copy(); init_env_kwargs.pop('initial_link_bandwidths', None)
    try: init_env = SFCEnv(sfc_requests_file, topology_file, **init_env_kwargs); initial_bandwidths = deepcopy(init_env.initial_link_resources); init_env.close()
    except Exception as e: print(f"Error creating initial env in make_sfc_env: {e}"); raise
    def env_creator():
        creator_kwargs = kwargs.copy(); creator_kwargs['initial_link_bandwidths'] = initial_bandwidths
        try: env = SFCEnv(sfc_requests_file, topology_file, **creator_kwargs); return env
        except Exception as e: print(f"Error creating env instance in env_creator: {e}"); raise
    if training_num > 0: train_envs = DummyVectorEnv([env_creator for _ in range(training_num)])
    else: train_envs = None
    if test_num > 0: test_envs = DummyVectorEnv([env_creator for _ in range(test_num)])
    else: test_envs = None
    single_env = env_creator()
    if train_envs: train_envs.seed(seed)
    if test_envs: test_envs.seed(seed + training_num)
    random.seed(seed); np.random.seed(seed)
    return single_env, train_envs, test_envs
