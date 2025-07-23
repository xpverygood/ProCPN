# --- START OF COMPLETE sfc_main.py (Fixed Space Access & InfoProcessorWrapper) ---

import os
from datetime import datetime
import gym # Use gym or gymnasium based on your installation
import torch
import torch.nn as nn
import numpy as np
import random
from typing import Any, Dict, List, Optional, Tuple, Union
from copy import deepcopy
import traceback # For detailed error printing

# Tianshou imports
from tianshou.data import Collector, VectorReplayBuffer, ReplayBuffer, Batch
from tianshou.trainer import onpolicy_trainer
from tianshou.utils import TensorboardLogger
from torch.utils.tensorboard import SummaryWriter
from tianshou.env import BaseVectorEnv # Needed for the Wrapper definition
import networkx as nx

# Project specific imports
# Ensure these imports work correctly based on your project structure
from sfc_env import SFCEnv, make_sfc_env
from model import MLP, ValueCritic
from diffusion import Diffusion
from sfc_policy import SFCDiffusionPPOPolicy

# --- Info Processor Wrapper Definition ---
class InfoProcessorWrapper(BaseVectorEnv):
    """
    A VectorEnv wrapper that processes the 'info' dictionary returned by step() and reset().
    It converts any numpy arrays/scalars within each info dict into Python native types
    to prevent Tianshou's Collector from failing during stacking due to
    inconsistent shapes/types across parallel environments.
    """
    def __init__(self, env: BaseVectorEnv):
        # Do NOT call super().__init__() as it might expect args like env_fns
        self.env = env
        if not hasattr(self.env, 'env_num'):
            raise AttributeError(f"The wrapped environment {type(env)} must have an 'env_num' attribute.")
        self._env_num = self.env.env_num

    def _process_info_list(self, info_list: List[Dict]) -> List[Dict]:
        """Helper function to process a list of info dictionaries."""
        processed_info_list = []
        for info_dict in info_list:
            processed_dict = {}
            if isinstance(info_dict, dict):
                for key, value in info_dict.items():
                    if isinstance(value, np.ndarray):
                        processed_dict[key] = value.tolist() # Convert array to list
                    elif isinstance(value, (np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, np.int64,
                                             np.uint8, np.uint16, np.uint32, np.uint64)):
                        processed_dict[key] = int(value) # Convert numpy int to Python int
                    elif isinstance(value, (np.float_, np.float16, np.float32, np.float64)):
                         processed_dict[key] = float(value) # Convert numpy float to Python float
                    elif isinstance(value, (np.bool_)):
                         processed_dict[key] = bool(value) # Convert numpy bool to Python bool
                    # Add specific handling for other numpy types if necessary
                    # elif isinstance(value, np.complex_):
                    #     processed_dict[key] = complex(value)
                    else:
                        # Keep standard Python types as they are
                        processed_dict[key] = value
            else:
                 processed_dict = info_dict # Pass through if not a dict
            processed_info_list.append(processed_dict)
        return processed_info_list

    def step(
        self,
        action: np.ndarray,
        id: Optional[Union[int, List[int], np.ndarray]] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[dict]]:
        """Steps the wrapped environments and processes the returned info."""
        obs, rew, terminated, truncated, info_list = self.env.step(action, id)
        processed_info_list = self._process_info_list(info_list)
        return obs, rew, terminated, truncated, processed_info_list

    def reset(
        self,
        id: Optional[Union[int, List[int], np.ndarray]] = None,
        seed: Optional[Union[int, List[int]]] = None,
        options: Optional[dict] = None
    ) -> Tuple[np.ndarray, List[dict]]:
        """Resets the wrapped environments and processes the returned info."""
        reset_kwargs = {}
        if seed is not None: reset_kwargs['seed'] = seed
        if options is not None: reset_kwargs['options'] = options

        try:
            # Tianshou DummyVectorEnv reset should return obs, info
            reset_result = self.env.reset(id=id, **reset_kwargs)
        except Exception as e:
             print(f"Error calling wrapped env's reset: {e}")
             traceback.print_exc()
             obs_space = self.env.observation_space
             # Use sample() if available, otherwise create zeros of correct shape/dtype
             if hasattr(obs_space, 'sample'):
                 dummy_obs = np.array([obs_space.sample() for _ in range(self.env_num)])
             else:
                 dummy_obs = np.zeros((self.env_num,) + obs_space.shape, dtype=obs_space.dtype)
             dummy_info = [{} for _ in range(self.env_num)]
             return dummy_obs, dummy_info

        if isinstance(reset_result, tuple) and len(reset_result) == 2:
            obs, info_list = reset_result
            if not isinstance(info_list, list) or len(info_list) != self.env_num:
                 # print(f"Warning: reset() returned info_list with unexpected type/length ({type(info_list)}, len={len(info_list) if isinstance(info_list, list) else 'N/A'}). Using empty info.")
                 info_list = [{} for _ in range(self.env_num)]
        elif isinstance(reset_result, np.ndarray):
            obs = reset_result
            info_list = [{} for _ in range(self.env_num)]
            print("Warning: Wrapped env reset() did not return info list. Using empty info.")
        else:
             raise TypeError(f"Unexpected return type from wrapped env reset(): {type(reset_result)}")

        processed_info_list = self._process_info_list(info_list)
        return obs, processed_info_list

    def seed(self, seed: Optional[Union[int, List[int]]] = None) -> List[Optional[List[int]]]:
        return self.env.seed(seed)

    def render(self, **kwargs) -> List[Any]:
        return self.env.render(**kwargs)

    def close(self) -> None:
        self.env.close()

    @property
    def env_num(self) -> int:
        return self._env_num

    @property
    def observation_space(self):
        # This property might still be problematic if self.env.observation_space is not standard.
        # However, we get the spaces from the unwrapped env in the main script now.
        return self.env.observation_space

    @property
    def action_space(self):
        # Same potential issue as observation_space.
        return self.env.action_space

    def __len__(self) -> int:
        return self.env_num

    def __getattr__(self, name: str) -> Any:
        if name.startswith('_') or name == 'env' or name in self.__dict__:
            return object.__getattribute__(self, name)
        try:
            return getattr(self.env, name)
        except AttributeError:
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}' and neither does the wrapped env '{type(self.env).__name__}'")

class EpochRewardLogger(TensorboardLogger):
    """Logs average test reward per epoch to a TXT file and TensorBoard."""
    def __init__(self, log_path: str, update_interval=100, info_interval=1000, save_interval=1):
        os.makedirs(log_path, exist_ok=True)
        writer = SummaryWriter(log_path)
        super().__init__(writer, update_interval, info_interval)
        self.log_txt_path = os.path.join(log_path, "epoch_test_rewards.txt")
        self.last_logged_epoch = -1
        self.log_txt_file = None
        try:
            self.log_txt_file = open(self.log_txt_path, "a")
            if os.path.getsize(self.log_txt_path) == 0:
                self.log_txt_file.write("Epoch\tEnvStep\tAvgTestReward\tStdTestReward\n")
                self.log_txt_file.flush()
        except IOError as e:
            print(f"Error opening log file {self.log_txt_path}: {e}")
            self.log_txt_file = None

    def log_test_data(self, collect_result: dict, step: int) -> None:
        super().log_test_data(collect_result, step)
        global STEP_PER_EPOCH
        if STEP_PER_EPOCH is None or STEP_PER_EPOCH <= 0:
             current_epoch = 0
        else:
             current_epoch = step // STEP_PER_EPOCH

        if self.log_txt_file is not None and current_epoch > self.last_logged_epoch:
            if "rews" in collect_result and isinstance(collect_result["rews"], np.ndarray) and collect_result["rews"].size > 0:
                avg_test_reward = collect_result["rews"].mean()
                std_test_reward = collect_result["rews"].std()
                try:
                    log_line = f"{current_epoch}\t{step}\t{avg_test_reward:.6f}\t{std_test_reward:.6f}\n"
                    self.log_txt_file.write(log_line)
                    self.log_txt_file.flush()
                    self.last_logged_epoch = current_epoch
                except IOError as e:
                    print(f"Error writing to test log file: {e}")

    def close(self) -> None:
        super().close()
        if self.log_txt_file is not None:
            try:
                print(f"Closing reward log file: {self.log_txt_path}")
                self.log_txt_file.close()
            except IOError as e:
                print(f"Error closing test log file: {e}")
            finally:
                 self.log_txt_file = None


# --- Configuration ---
SFC_REQUEST_FILE = "data/sfc4.json"
TOPOLOGY_FILE = "data/Chinanet.graphml"
LOG_DIR = "log_sfc_diffusion_ppo"
NODE_CPU_CAPACITY = 10.0
MAX_ALLOWED_INSTANCES = 50
DECISION_SCHEDULE = 'sfc_polling'
REWARD_WEIGHTS = {'accept': 5.0, 'deploy': 0.25, 'hops': 0.05}
INVALID_ACTION_PENALTY = -20.0
VNF_INSTANCE_CAPACITY = 5
K_SHORTEST_PATHS = 4
HIDDEN_DIM = 256
T_DIM = 16
ACTIVATION = 'relu'
N_TIMESTEPS = 5
BETA_SCHEDULE = 'linear'
LOSS_TYPE = 'l2'
BC_COEF = 0.0
LEARNING_RATE = 3e-5
GAMMA = 0.99
GAE_LAMBDA = 0.85
VF_COEF = 0.25
ENT_COEF = 0.01
EPS_CLIP = 0.15
ADV_NORM = True
RECOMPUTE_ADV = False
GRAD_NORM = 0.5
SEED = 1
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TRAINING_ENV_NUM = 4
TEST_ENV_NUM = 1
BUFFER_SIZE = 20000
EPOCH = 150
STEP_PER_EPOCH = 30000
STEP_PER_COLLECT = 2048
REPEAT_PER_COLLECT = 4
BATCH_SIZE = 128
EPISODE_PER_TEST = 10
N_FINAL_DETAILED_RUNS = 20

print(f"--- Setting Seed: {SEED} ---")
np.random.seed(SEED)
random.seed(SEED)
torch.manual_seed(SEED)
if DEVICE == "cuda":
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

print("--- Setting up Environment ---")
env_kwargs = {
    'default_node_cpu': NODE_CPU_CAPACITY,
    'max_allowed_instances': MAX_ALLOWED_INSTANCES,
    'decision_schedule_type': DECISION_SCHEDULE,
    'reward_weights': REWARD_WEIGHTS,
    'invalid_action_penalty': INVALID_ACTION_PENALTY,
    'vnf_instance_capacity': VNF_INSTANCE_CAPACITY,
    'k_shortest_paths': K_SHORTEST_PATHS
}
_env, _train_envs, _test_envs = make_sfc_env(
    sfc_requests_file=SFC_REQUEST_FILE,
    topology_file=TOPOLOGY_FILE,
    training_num=TRAINING_ENV_NUM,
    test_num=TEST_ENV_NUM,
    seed=SEED,
    **env_kwargs
)

print("--- Getting Environment Spaces ---")
if _env is None:
    raise RuntimeError("Failed to create the single environment instance (_env). Cannot determine spaces.")

try:
    observation_space = _env.observation_space
    action_space = _env.action_space

    # Validate the spaces retrieved
    if not hasattr(observation_space, 'shape'):
        raise TypeError(f"Retrieved observation_space is not a valid Gym space with a shape attribute. Type: {type(observation_space)}")
    if not hasattr(action_space, 'n'):
         raise TypeError(f"Retrieved action_space is not a valid Gym Discrete space with an 'n' attribute. Type: {type(action_space)}")

    state_shape = observation_space.shape
    state_dim = observation_space.shape[0]
    action_shape = action_space.n # For Discrete space
    action_dim = action_space.n

    print(f"Observation space shape: {state_shape}, Dim: {state_dim}")
    print(f"Action space shape: {action_shape}, Dim: {action_dim} (Number of Nodes)")

except Exception as e:
    print(f"Error accessing spaces from _env: {e}")
    traceback.print_exc()
    raise RuntimeError("Could not correctly determine observation/action space from the environment.")


print("--- Applying InfoProcessorWrapper ---")
env = _env # Keep single env unwrapped
train_envs = InfoProcessorWrapper(_train_envs) if _train_envs is not None else None
test_envs = InfoProcessorWrapper(_test_envs) if _test_envs is not None else None
print("Vector environments wrapped.")

if env is not None and hasattr(env, 'initial_link_resources'):
    initial_bandwidths_for_eval = deepcopy(env.initial_link_resources)
else:
    print("Warning: Could not get initial bandwidths from single env instance.")
    initial_bandwidths_for_eval = None


print(f"--- Setting up Models (Device: {DEVICE}) ---")
# Use the state_dim and action_dim obtained reliably above
diffusion_mlp = MLP(
    state_dim=state_dim, action_dim=action_dim, hidden_dim=HIDDEN_DIM,
    t_dim=T_DIM, activation=ACTIVATION
).to(DEVICE)
diffusion_model = Diffusion(
    state_dim=state_dim, action_dim=action_dim, model=diffusion_mlp,
    beta_schedule=BETA_SCHEDULE, n_timesteps=N_TIMESTEPS, loss_type=LOSS_TYPE,
    bc_coef=BC_COEF
).to(DEVICE)
critic = ValueCritic(
    state_dim=state_dim, hidden_dim=HIDDEN_DIM, activation=ACTIVATION
).to(DEVICE)
print("Models created.")

# --- Setup Optimizer ---
print("--- Setting up Optimizer ---")
actor_lr = LEARNING_RATE
critic_lr = 1e-4
optim = torch.optim.Adam([
    {'params': diffusion_model.model.parameters(), 'lr': actor_lr},
    {'params': critic.parameters(), 'lr': critic_lr}
])
print(f"Optimizer Adam set up (Actor LR: {actor_lr}, Critic LR: {critic_lr})")

print("--- Setting up Policy ---")
policy = SFCDiffusionPPOPolicy(
    diffusion_model=diffusion_model, critic=critic, optim=optim,
    action_space=action_space, # Pass the retrieved action_space
    eps_clip=EPS_CLIP, advantage_normalization=ADV_NORM, recompute_advantage=RECOMPUTE_ADV,
    vf_coef=VF_COEF, ent_coef=ENT_COEF, gamma=GAMMA, gae_lambda=GAE_LAMBDA,
    reward_normalization=False, max_grad_norm=GRAD_NORM
).to(DEVICE)
print("SFCDiffusionPPOPolicy created.")

# --- Setup Collector ---
print("--- Setting up Collectors ---")
train_collector = None
test_collector = None

if train_envs is not None:
    train_replay_buffer = VectorReplayBuffer(
        total_size=BUFFER_SIZE, buffer_num=train_envs.env_num
    )
    print(f"VectorReplayBuffer created (Size: {BUFFER_SIZE}, Num Envs: {train_envs.env_num})")
    train_collector = Collector(policy=policy, env=train_envs, buffer=train_replay_buffer)
    print("Train Collector created (using wrapped envs).")
else:
    print("Skipping Train Collector setup (train_envs is None).")

if test_envs is not None:
    test_collector = Collector(policy=policy, env=test_envs)
    print("Test Collector created (using wrapped envs).")
else:
    print("Skipping Test Collector setup (test_envs is None).")


print("--- Setting up Logger ---")
log_name = datetime.now().strftime('%y%m%d-%H%M%S') + f"_seed{SEED}"
log_path = os.path.join(LOG_DIR, log_name)
logger = EpochRewardLogger(log_path)
writer = logger.writer
config_dict = {
    'env_kwargs': env_kwargs, 'k_shortest_paths': K_SHORTEST_PATHS,
    'actor_lr': actor_lr, 'critic_lr': critic_lr, 'hidden_dim': HIDDEN_DIM,
    't_dim': T_DIM, 'activation': ACTIVATION, 'n_timesteps': N_TIMESTEPS,
    'beta_schedule': BETA_SCHEDULE, 'gamma': GAMMA, 'gae_lambda': GAE_LAMBDA,
    'vf_coef': VF_COEF, 'ent_coef': ENT_COEF, 'eps_clip': EPS_CLIP,
    'adv_norm': ADV_NORM, 'grad_norm': GRAD_NORM, 'seed': SEED, 'device': DEVICE,
    'epoch': EPOCH, 'step_per_epoch': STEP_PER_EPOCH, 'step_per_collect': STEP_PER_COLLECT,
    'repeat_per_collect': REPEAT_PER_COLLECT, 'batch_size': BATCH_SIZE
}
config_str = "\n".join([f"{k}: {v}" for k, v in config_dict.items()])
config_str_safe = config_str.replace('{','(').replace('}',')').replace(': ','=')
writer.add_text("config", config_str_safe)
print(f"Logging enabled. Log directory: {log_path}")

print("--- Setting up Trainer Callbacks ---")
def stop_fn(mean_rewards): return False
def save_best_fn(policy):
    model_save_path = os.path.join(log_path, "best_policy.pth")
    try:
        torch.save(policy.state_dict(), model_save_path)
        print(f"\nBest policy saved to {model_save_path}")
    except Exception as e: print(f"\nError saving best policy: {e}")
def save_checkpoint_fn(epoch, env_step, gradient_step):
    ckpt_path = os.path.join(log_path, f"checkpoint_epoch{epoch}.pth")
    try:
        torch.save({'policy': policy.state_dict(), 'optim': optim.state_dict()}, ckpt_path)
    except Exception as e: print(f"Error saving checkpoint: {e}")
    return ckpt_path
print("Callbacks defined.")

# --- Start Training ---
print("\n--- Starting Training ---")
if train_collector is None:
     print("\nERROR: Train collector is None. Aborting training.")
     exit()

try:
    result = onpolicy_trainer(
        policy=policy,
        train_collector=train_collector,
        test_collector=test_collector, # Can be None
        max_epoch=EPOCH,
        step_per_epoch=STEP_PER_EPOCH,
        repeat_per_collect=REPEAT_PER_COLLECT,
        episode_per_test=EPISODE_PER_TEST,
        batch_size=BATCH_SIZE,
        step_per_collect=STEP_PER_COLLECT,
        stop_fn=stop_fn,
        save_best_fn=save_best_fn,
        logger=logger,
        test_in_train= (test_collector is not None),
        save_checkpoint_fn=save_checkpoint_fn
    )
    print("\n--- Training Finished Successfully ---")
    print(f"Final Training Stats: {result}")

except Exception as e:
     print("\n--- Training Interrupted by Exception ---")
     print(f"Error Type: {type(e).__name__}")
     print(f"Error Details: {e}")
     traceback.print_exc()
     error_save_path = os.path.join(log_path, "policy_on_error.pth")
     try:
        torch.save(policy.state_dict(), error_save_path)
        print(f"Policy state saved to {error_save_path}")
     except Exception as save_e: print(f"Could not save policy state on error: {save_e}")


# --- Final Evaluation ---
print("\n--- Starting Final Evaluation ---")
best_policy_path = os.path.join(log_path, "best_policy.pth")
if os.path.exists(best_policy_path):
    try:
        policy.load_state_dict(torch.load(best_policy_path, map_location=DEVICE))
        print(f"Loaded best policy from {best_policy_path}")
    except Exception as e: print(f"Error loading best policy: {e}. Using final state.")
else:
    print(f"Warning: best_policy.pth not found. Using final policy state.")

policy.eval()

if test_collector:
    print("Running final evaluation using test_collector...")
    try:
        final_result = test_collector.collect(n_episode=50, render=0.0)
        final_rewards = final_result.get("rews")
        if isinstance(final_rewards, np.ndarray) and final_rewards.size > 0:
            avg_reward = final_rewards.mean()
            std_reward = final_rewards.std()
            num_eval_episodes = final_result.get("n/ep", len(final_rewards))
            print("-" * 40)
            print("Final Evaluation Results (Loaded Policy):")
            print(f"  Average Reward (over {num_eval_episodes} episodes): {avg_reward:.4f} +/- {std_reward:.4f}")
            print("-" * 40)
        else: print(f"Final evaluation reward data invalid or missing. Result keys: {final_result.keys()}")
    except Exception as e:
         print(f"Error during final evaluation collection: {e}")
         traceback.print_exc()
else:
    print("Skipping final evaluation (test_collector is None).")


# --- Find and Output Best Single Run Strategy ---
print(f"\n--- Running {N_FINAL_DETAILED_RUNS} detailed episodes ---")
policy.eval()

best_run_reward = -float('inf')
best_run_sfc_states = None
all_run_rewards = []

# Use the single UNWRAPPED env (_env) for detailed evaluation
if _env is None:
    print("ERROR: Single environment instance (_env) is None. Cannot run detailed evaluation.")
elif initial_bandwidths_for_eval is None:
     print("ERROR: Initial bandwidths not stored. Cannot run consistent detailed evaluation.")
else:
    print("Recreating single UNWRAPPED evaluation env with consistent initial bandwidths...")
    eval_env_kwargs = env_kwargs.copy()
    eval_env_kwargs['initial_link_bandwidths'] = initial_bandwidths_for_eval
    try:
        single_eval_env = SFCEnv(SFC_REQUEST_FILE, TOPOLOGY_FILE, **eval_env_kwargs)
    except Exception as e:
        print(f"ERROR: Could not recreate single eval env: {e}")
        single_eval_env = None

    if single_eval_env:
        eval_env_seed_base = SEED + 2000
        print(f"Starting {N_FINAL_DETAILED_RUNS} detailed runs...")
        for run_idx in range(N_FINAL_DETAILED_RUNS):
            current_seed = eval_env_seed_base + run_idx
            try:
                obs, info = single_eval_env.reset(seed=current_seed)
                terminated, truncated = False, False
                episode_reward = 0.0
                step = 0
                max_steps = len(single_eval_env.decision_queue) * 2 if single_eval_env.decision_queue else 500

                while not (terminated or truncated):
                    obs_batch = Batch(obs=np.expand_dims(obs, axis=0), info={})
                    with torch.no_grad(): policy_result = policy(obs_batch)
                    act = policy_result.act[0].item()
                    obs, reward, terminated, truncated, info = single_eval_env.step(act)
                    episode_reward += reward
                    step += 1
                    if step > max_steps: truncated = True; print(f"Eval run {run_idx+1} truncated (max steps).")

                all_run_rewards.append(episode_reward)
                if episode_reward > best_run_reward:
                    best_run_reward = episode_reward
                    best_run_sfc_states = deepcopy(single_eval_env.sfc_states)

            except Exception as e: print(f"\nError during detailed run {run_idx + 1}: {e}"); traceback.print_exc()

        print(f"\nCompleted {len(all_run_rewards)} detailed runs.")
        if all_run_rewards: print(f"Avg reward: {np.mean(all_run_rewards):.4f} +/- {np.std(all_run_rewards):.4f}")
        else: print("No detailed runs completed.")

        # Print best strategy
        print("\n" + "-" * 40 + "\n--- Best Found Strategy ---")
        if best_run_sfc_states is not None:
            print(f"Best single episode reward: {best_run_reward:.4f}")
            sfc_requests_map = {req['sfc_id']: req for req in single_eval_env.sfc_requests}
            for state in best_run_sfc_states:
                sfc_req = sfc_requests_map.get(state.get('id'))
                print("-" * 20)
                if sfc_req:
                    print(f"SFC ID: {state['id']} (Req: {sfc_req['source_node']}->{sfc_req['destination_node']}, BW: {sfc_req.get('bandwidth')})")
                    print(f"  Status: {state['status']}")
                    if state['status'].startswith('failed'): print(f"  Fail Reason: {state.get('fail_reason')}")
                    print(f"  VNF Seq Req: {sfc_req['vnf_seq']}")
                    placement_nodes = state.get('vnf_nodes', [])[1:]
                    print(f"  Placement: {placement_nodes}")
                    if state['status'] == 'success':
                        path = state.get('routing_path', [])
                        hops = state.get('total_hops', 0)
                        print(f"  Route Path: {path}")
                        print(f"  Route Hops: {hops}")
                    else: print("  Routing: Failed or Incomplete")
                else: print(f"SFC ID: {state.get('id')} - Request details missing.")
        else: print("Could not determine a best run strategy.")
        print("-" * 40)

        try: single_eval_env.close()
        except Exception as e: print(f"Error closing recreated single_eval_env: {e}")

    else: print("Skipping detailed evaluation runs (env creation failed).")


# --- Cleanup ---
print("\n--- Cleaning up ---")
# Close WRAPPED vector envs
if train_envs is not None:
    try: train_envs.close()
    except Exception as e: print(f"Error closing wrapped train_envs: {e}")
if test_envs is not None:
     try: test_envs.close()
     except Exception as e: print(f"Error closing wrapped test_envs: {e}")
if env is not None: # This is _env
     try: env.close()
     except Exception as e: print(f"Error closing unwrapped single env ref (_env): {e}")
# Close original vector envs (may be redundant if wrapper closes them, but safe)
if '_train_envs' in locals() and _train_envs is not None:
    try: _train_envs.close()
    except Exception as e: print(f"Error closing original _train_envs: {e}")
if '_test_envs' in locals() and _test_envs is not None:
     try: _test_envs.close()
     except Exception as e: print(f"Error closing original _test_envs: {e}")

if logger is not None:
     try: logger.close()
     except Exception as e: print(f"Error closing logger: {e}")
print("Script finished.")
