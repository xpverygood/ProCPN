from typing import Any, Dict, List, Type, Optional, Union

import numpy as np
import torch
from torch import nn
from copy import deepcopy

from tianshou.data import Batch, ReplayBuffer, to_torch, to_torch_as
from tianshou.policy import PPOPolicy
from tianshou.utils.net.common import ActorCritic
from torch.distributions.categorical import Categorical

class TelemetryDiff(PPOPolicy):
    def __init__(
        self,
        actor: torch.nn.Module,
        critic: torch.nn.Module,
        optim: torch.optim.Optimizer,
        env,
        device: torch.device,
        dist_fn: Type[torch.distributions.Distribution],
        eps_clip: float = 0.2,
        dual_clip: Optional[float] = None,
        value_clip: bool = False,
        advantage_normalization: bool = True,
        recompute_advantage: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(actor, critic, optim, dist_fn, **kwargs)
        self._eps_clip = eps_clip
        assert dual_clip is None or dual_clip > 1.0, \
            "Dual-clip PPO parameter should greater than 1.0."
        self._dual_clip = dual_clip
        self._value_clip = value_clip
        self._norm_adv = advantage_normalization
        self._recompute_adv = recompute_advantage
        self._actor_critic: ActorCritic

        self._actor = actor

        self._device = device  # Device to run computations on
        self.env = env
        self.edge_index = deepcopy(self.env.edge_index)
        self.edge_index = self.edge_index.to(device)

    def forward(
            self,
            batch: Batch,
            state: Optional[Union[dict, Batch, np.ndarray]] = None,
            input: str = "obs",
            model: str = "actor"
    ) -> Batch:
        # Convert batch observations to PyTorch tensors
        obs_ = to_torch(batch[input], device=self._device, dtype=torch.float32)
        mask = to_torch(batch['info'].mask, device=self._device)
        # Use actor or target actor based on provided model argument
        model_ = self._actor
        # Feed observations through the selected model to get action logits
        
        logits = model_.get_logits(obs_, self.edge_index)
        pi = Categorical(logits=logits)
        mask = torch.squeeze(mask)
        logits_delta = torch.zeros(mask.size()).to(mask.device)
        logits_delta[mask == 0] = float("-Inf")
        logits_ = logits + logits_delta
        pi_mask = Categorical(logits = logits_)
        
        a = pi_mask.sample()
        
        # noise = to_torch(self.noise_generator.generate(logits.shape),
        #                  dtype=torch.float32, device=self._device)
        # Add the noise to the action
        # acts = logits + noise
        acts = a
        # acts = torch.clamp(noisy_action, -1, 1)
        dist = pi  # does not use a probability distribution for actions

        return Batch(logits=logits, act=acts, state=obs_, dist=dist)
    
    def process_fn(
        self, batch: Batch, buffer: ReplayBuffer, indices: np.ndarray
    ) -> Batch:
        if self._recompute_adv:
            # buffer input `buffer` and `indices` to be used in `learn()`.
            self._buffer, self._indices = buffer, indices
        batch = self._compute_returns(batch, buffer, indices)
        batch.act = to_torch_as(batch.act, batch.v_s)
        with torch.no_grad():
            batch.logp_old = self(batch).dist.log_prob(batch.act)
        return batch
    
    def _compute_returns(
        self, batch: Batch, buffer: ReplayBuffer, indices: np.ndarray
    ) -> Batch:
        v_s, v_s_ = [], []
        with torch.no_grad():
            for minibatch in batch.split(self._batch, shuffle=False, merge_last=True):
                obs_tem = to_torch(minibatch.obs, device=self._device, dtype=torch.float32)
                obs_next_tem = to_torch(minibatch.obs, device=self._device, dtype=torch.float32)
                v_s.append(self.critic(obs_tem, self.edge_index))
                v_s_.append(self.critic(obs_next_tem, self.edge_index))
        batch.v_s = torch.cat(v_s, dim=0).flatten()  # old value
        v_s = batch.v_s.cpu().numpy()
        v_s_ = torch.cat(v_s_, dim=0).flatten().cpu().numpy()
        # when normalizing values, we do not minus self.ret_rms.mean to be numerically
        # consistent with OPENAI baselines' value normalization pipeline. Emperical
        # study also shows that "minus mean" will harm performances a tiny little bit
        # due to unknown reasons (on Mujoco envs, not confident, though).
        if self._rew_norm:  # unnormalize v_s & v_s_
            v_s = v_s * np.sqrt(self.ret_rms.var + self._eps)
            v_s_ = v_s_ * np.sqrt(self.ret_rms.var + self._eps)
        unnormalized_returns, advantages = self.compute_episodic_return(
            batch,
            buffer,
            indices,
            v_s_,
            v_s,
            gamma=self._gamma,
            gae_lambda=self._lambda
        )
        if self._rew_norm:
            batch.returns = unnormalized_returns / \
                np.sqrt(self.ret_rms.var + self._eps)
            self.ret_rms.update(unnormalized_returns)
        else:
            batch.returns = unnormalized_returns
        batch.returns = to_torch_as(batch.returns, batch.v_s)
        batch.adv = to_torch_as(advantages, batch.v_s)
        return batch

    def learn(  # type: ignore
            self, batch: Batch, batch_size: int, repeat: int, **kwargs: Any
        ) -> Dict[str, List[float]]:
            losses, clip_losses, vf_losses, ent_losses = [], [], [], []
            for step in range(repeat):
                if self._recompute_adv and step > 0:
                    batch = self._compute_returns(batch, self._buffer, self._indices)
                for minibatch in batch.split(batch_size, merge_last=True):
                    # calculate loss for actor
                    dist = self(minibatch).dist
                    if self._norm_adv:
                        mean, std = minibatch.adv.mean(), minibatch.adv.std()
                        minibatch.adv = (minibatch.adv -
                                        mean) / (std + self._eps)  # per-batch norm
                    ratio = (dist.log_prob(minibatch.act) - minibatch.logp_old).exp().float()
                    ratio = ratio.reshape(ratio.size(0), -1).transpose(0, 1)
                    surr1 = ratio * minibatch.adv
                    surr2 = ratio.clamp(
                        1.0 - self._eps_clip, 1.0 + self._eps_clip
                    ) * minibatch.adv
                    if self._dual_clip:
                        clip1 = torch.min(surr1, surr2)
                        clip2 = torch.max(clip1, self._dual_clip * minibatch.adv)
                        clip_loss = -torch.where(minibatch.adv < 0, clip2, clip1).mean()
                    else:
                        clip_loss = -torch.min(surr1, surr2).mean()
                    # calculate loss for critic
                    obs_tem = to_torch(minibatch.obs, device=self._device, dtype=torch.float32)
                    value = self.critic(obs_tem, self.edge_index).flatten()
                    if self._value_clip:
                        v_clip = minibatch.v_s + \
                            (value - minibatch.v_s).clamp(-self._eps_clip, self._eps_clip)
                        vf1 = (minibatch.returns - value).pow(2)
                        vf2 = (minibatch.returns - v_clip).pow(2)
                        vf_loss = torch.max(vf1, vf2).mean()
                    else:
                        vf_loss = (minibatch.returns - value).pow(2).mean()
                    # calculate regularization and overall loss
                    ent_loss = dist.entropy().mean()
                    loss = clip_loss + self._weight_vf * vf_loss \
                        - self._weight_ent * ent_loss
                    self.optim.zero_grad()
                    loss.backward()
                    if self._grad_norm:  # clip large gradient
                        nn.utils.clip_grad_norm_(
                            self._actor_critic.parameters(), max_norm=self._grad_norm
                        )
                    self.optim.step()
                    clip_losses.append(clip_loss.item())
                    vf_losses.append(vf_loss.item())
                    ent_losses.append(ent_loss.item())
                    losses.append(loss.item())

            return {
                "loss": losses,
                "loss/clip": clip_losses,
                "loss/vf": vf_losses,
                "loss/ent": ent_losses,
            }
