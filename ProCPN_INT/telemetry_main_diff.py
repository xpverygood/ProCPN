from datetime import datetime
import os
import sys
import gymnasium as gym
import torch
import torch.nn as nn

from tianshou.data import Collector, VectorReplayBuffer
from tianshou.env import DummyVectorEnv
from tianshou.policy import PPOPolicy
from tianshou.trainer import OnpolicyTrainer
from tianshou.utils.net.common import ActorCritic, Net
from tianshou.utils.net.discrete import Actor, Critic
from torch.utils.tensorboard import SummaryWriter
from tianshou.utils import TensorboardLogger

from env.network_env import make_telemetry_env
from telemetry_diffusion.gcndiff import GCNCategoricalActor, GCNCritic, GCN
from telemetry_diffusion.telemetry_policy_diff import TelemetryDiff
from env.path_collector import path_Collector

device = "cuda" if torch.cuda.is_available() else "cpu"

# environments
env, train_envs, test_envs = make_telemetry_env(1, 1)

# seed = 1

# if torch.cuda.is_available():
#     torch.cuda.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed)

node_num = env.observation_space.shape[0]
feature_num = env.observation_space.shape[1]
hidden_sizes = (64, 64)
activation = nn.ReLU

state_shape = env.observation_space.shape
action_shape = env.action_space.n

# create gcn
gcn = GCN(feature_num, feature_num)
    
# Actor is a Diffusion model
actor = GCNCategoricalActor(
    feature_num=feature_num,
    node_num=node_num,
    gcn=gcn,
    hidden_sizes=hidden_sizes,
    act_num=action_shape,
    activation=activation,
).to(device)

# Create critic
critic = GCNCritic(
    feature_num=feature_num,
    node_num=node_num,
    gcn=gcn,
    hidden_sizes=hidden_sizes,
    activation=activation
).to(device)

actor_critic = ActorCritic(actor, critic)
optim = torch.optim.Adam(actor_critic.parameters(), lr=0.0003)

## Setup logging
time_now = datetime.now().strftime('%b%d-%H%M%S')
log_path = os.path.join('log_diff', time_now)
writer = SummaryWriter(log_path)
writer.add_text("args", 'test')
logger = TensorboardLogger(writer)

#policy
dist = torch.distributions.Categorical
policy = TelemetryDiff(
    actor=actor,
    critic=critic,
    optim=optim,
    env=env,
    device=device,
    dist_fn=dist,
    action_space=env.action_space,
    action_scaling=False,
)

# collector
train_collector = Collector(policy, train_envs, VectorReplayBuffer(20000, len(train_envs)))
test_collector = Collector(policy, test_envs)

# trainer
result = OnpolicyTrainer(
    policy=policy,
    batch_size=256,
    train_collector=train_collector,
    test_collector=test_collector,
    max_epoch=200,
    step_per_epoch=1,
    repeat_per_collect=10,
    episode_per_test=10,
    step_per_collect=1000,
    logger = logger,
).run()
print(result)

# Let's watch its performance!
policy.eval()

probe_path_collector = path_Collector(policy, test_envs)
probe_path = probe_path_collector.collect(n_episode=1)
print("Final reward: {}, length: {}".format(probe_path["rews"].mean(), probe_path["lens"].mean()))
print(probe_path["probe_path"])
