import time
import warnings
from typing import Any, Callable, Dict, List, Optional, Union

import gym
import numpy as np
import torch

from tianshou.data import (
    Batch,
    CachedReplayBuffer,
    ReplayBuffer,
    ReplayBufferManager,
    VectorReplayBuffer,
    to_numpy,
)
from tianshou.data.batch import _alloc_by_keys_diff
from tianshou.env import BaseVectorEnv, DummyVectorEnv
from tianshou.policy import BasePolicy

from tianshou.data import Collector

class path_Collector(Collector):
    def __init__(
        self,
        policy: BasePolicy,
        env: Union[gym.Env, BaseVectorEnv],
        buffer: Optional[ReplayBuffer] = None,
        preprocess_fn: Optional[Callable[..., Batch]] = None,
        exploration_noise: bool = False,
    ) -> None:
        super().__init__(policy, env)
        if isinstance(env, gym.Env) and not hasattr(env, "__len__"):
            warnings.warn("Single environment detected, wrap to DummyVectorEnv.")
            self.env = DummyVectorEnv([lambda: env])  # type: ignore
        else:
            self.env = env  # type: ignore
        self.env_num = len(self.env)
        self.exploration_noise = exploration_noise
        self._assign_buffer(buffer)
        self.policy = policy
        self.preprocess_fn = preprocess_fn
        self._action_space = self.env.action_space
        # avoid creating attribute outside __init__
        self.reset(False)
        
    def collect(
        self,
        n_step: Optional[int] = None,
        n_episode: Optional[int] = None,
        random: bool = False,
        render: Optional[float] = None,
        no_grad: bool = True,
        gym_reset_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Collect a specified number of step or episode.

        To ensure unbiased sampling result with n_episode option, this function will
        first collect ``n_episode - env_num`` episodes, then for the last ``env_num``
        episodes, they will be collected evenly from each env.

        :param int n_step: how many steps you want to collect.
        :param int n_episode: how many episodes you want to collect.
        :param bool random: whether to use random policy for collecting data. Default
            to False.
        :param float render: the sleep time between rendering consecutive frames.
            Default to None (no rendering).
        :param bool no_grad: whether to retain gradient in policy.forward(). Default to
            True (no gradient retaining).
        :param gym_reset_kwargs: extra keyword arguments to pass into the environment's
            reset function. Defaults to None (extra keyword arguments)

        .. note::

            One and only one collection number specification is permitted, either
            ``n_step`` or ``n_episode``.

        :return: A dict including the following keys

            * ``n/ep`` collected number of episodes.
            * ``n/st`` collected number of steps.
            * ``rews`` array of episode reward over collected episodes.
            * ``lens`` array of episode length over collected episodes.
            * ``idxs`` array of episode start index in buffer over collected episodes.
            * ``rew`` mean of episodic rewards.
            * ``len`` mean of episodic lengths.
            * ``rew_std`` standard error of episodic rewards.
            * ``len_std`` standard error of episodic lengths.
        """
        assert not self.env.is_async, "Please use AsyncCollector if using async venv."
        if n_step is not None:
            assert n_episode is None, (
                f"Only one of n_step or n_episode is allowed in Collector."
                f"collect, got n_step={n_step}, n_episode={n_episode}."
            )
            assert n_step > 0
            if not n_step % self.env_num == 0:
                warnings.warn(
                    f"n_step={n_step} is not a multiple of #env ({self.env_num}), "
                    "which may cause extra transitions collected into the buffer."
                )
            ready_env_ids = np.arange(self.env_num)
        elif n_episode is not None:
            assert n_episode > 0
            ready_env_ids = np.arange(min(self.env_num, n_episode))
            self.data = self.data[:min(self.env_num, n_episode)]
        else:
            raise TypeError(
                "Please specify at least one (either n_step or n_episode) "
                "in AsyncCollector.collect()."
            )

        start_time = time.time()

        step_count = 0
        episode_count = 0
        episode_rews = []
        episode_lens = []
        episode_start_indices = []

        while True:
            assert len(self.data) == len(ready_env_ids)
            # restore the state: if the last state is None, it won't store
            last_state = self.data.policy.pop("hidden_state", None)

            # get the next action
            if random:
                try:
                    act_sample = [
                        self._action_space[i].sample() for i in ready_env_ids
                    ]
                except TypeError:  # envpool's action space is not for per-env
                    act_sample = [self._action_space.sample() for _ in ready_env_ids]
                act_sample = self.policy.map_action_inverse(act_sample)  # type: ignore
                self.data.update(act=act_sample)
            else:
                if no_grad:
                    with torch.no_grad():  # faster than retain_grad version
                        # self.data.obs will be used by agent to get result
                        result = self.policy(self.data, last_state)
                else:
                    result = self.policy(self.data, last_state)
                # update state / act / policy into self.data
                policy = result.get("policy", Batch())
                assert isinstance(policy, Batch)
                state = result.get("state", None)
                if state is not None:
                    policy.hidden_state = state  # save state into buffer
                act = to_numpy(result.act)
                if self.exploration_noise:
                    act = self.policy.exploration_noise(act, self.data)
                self.data.update(policy=policy, act=act)

            # get bounded and remapped actions first (not saved into buffer)
            action_remap = self.policy.map_action(self.data.act)
            # step in env
            result = self.env.step(action_remap, ready_env_ids)  # type: ignore
            if len(result) == 5:
                obs_next, rew, terminated, truncated, info = result
                done = np.logical_or(terminated, truncated)
            elif len(result) == 4:
                obs_next, rew, done, info = result
                if isinstance(info, dict):
                    truncated = info["TimeLimit.truncated"]
                else:
                    truncated = np.array(
                        [
                            info_item.get("TimeLimit.truncated", False)
                            for info_item in info
                        ]
                    )
                terminated = np.logical_and(done, ~truncated)
            else:
                raise ValueError()

            self.data.update(
                obs_next=obs_next,
                rew=rew,
                terminated=terminated,
                truncated=truncated,
                done=done,
                info=info
            )
            if self.preprocess_fn:
                self.data.update(
                    self.preprocess_fn(
                        obs_next=self.data.obs_next,
                        rew=self.data.rew,
                        done=self.data.done,
                        info=self.data.info,
                        policy=self.data.policy,
                        env_id=ready_env_ids,
                    )
                )

            if render:
                self.env.render()
                if render > 0 and not np.isclose(render, 0):
                    time.sleep(render)

            # add data into the buffer
            ptr, ep_rew, ep_len, ep_idx = self.buffer.add(
                self.data, buffer_ids=ready_env_ids
            )

            # collect statistics
            step_count += len(ready_env_ids)

            if np.any(done):
                env_ind_local = np.where(done)[0]
                env_ind_global = ready_env_ids[env_ind_local]
                episode_count += len(env_ind_local)
                episode_lens.append(ep_len[env_ind_local])
                episode_rews.append(ep_rew[env_ind_local])
                episode_start_indices.append(ep_idx[env_ind_local])
                
                if max(episode_rews) == episode_rews[-1]:   
                    probe_path = self.env.workers[0].env._laststate["probe"]
                
                # now we copy obs_next to obs, but since there might be
                # finished episodes, we have to reset finished envs first.
                self._reset_env_with_ids(
                    env_ind_local, env_ind_global, gym_reset_kwargs
                )
                for i in env_ind_local:
                    self._reset_state(i)

                # remove surplus env id from ready_env_ids
                # to avoid bias in selecting environments
                if n_episode:
                    surplus_env_num = len(ready_env_ids) - (n_episode - episode_count)
                    if surplus_env_num > 0:
                        mask = np.ones_like(ready_env_ids, dtype=bool)
                        mask[env_ind_local[:surplus_env_num]] = False
                        ready_env_ids = ready_env_ids[mask]
                        self.data = self.data[mask]

            self.data.obs = self.data.obs_next

            if (n_step and step_count >= n_step) or \
                    (n_episode and episode_count >= n_episode):
                break

        # generate statistics
        self.collect_step += step_count
        self.collect_episode += episode_count
        self.collect_time += max(time.time() - start_time, 1e-9)

        if n_episode:
            self.data = Batch(
                obs={},
                act={},
                rew={},
                terminated={},
                truncated={},
                done={},
                obs_next={},
                info={},
                policy={}
            )
            self.reset_env()

        if episode_count > 0:
            rews, lens, idxs = list(
                map(
                    np.concatenate,
                    [episode_rews, episode_lens, episode_start_indices]
                )
            )
            rew_mean, rew_std = rews.mean(), rews.std()
            len_mean, len_std = lens.mean(), lens.std()
        else:
            rews, lens, idxs = np.array([]), np.array([], int), np.array([], int)
            rew_mean = rew_std = len_mean = len_std = 0

        return {
            "n/ep": episode_count,
            "n/st": step_count,
            "rews": rews,
            "lens": lens,
            "idxs": idxs,
            "rew": rew_mean,
            "len": len_mean,
            "rew_std": rew_std,
            "len_std": len_std,
            "probe_path": probe_path,
        } 