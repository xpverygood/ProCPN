
from typing import Any, Dict, List, Type, Optional, Union

import numpy as np
import torch
from torch import nn
from copy import deepcopy
import gym # Use gym
import random

from tianshou.data import Batch, ReplayBuffer, VectorReplayBuffer, to_torch, to_torch_as
from tianshou.policy import PPOPolicy
from torch.distributions import Categorical

from diffusion import Diffusion
from model import MLP, ValueCritic

class SFCDiffusionPPOPolicy(PPOPolicy):
    """
    PPO Policy using a Diffusion Model for VNF Placement (Node Selection).
    Assumes environment handles reuse and invalid actions via penalties. NO MASKING used.
    """
    def __init__(
        self,
        diffusion_model: Diffusion, # Diffusion model's action_dim should match env.action_space.n (num_nodes)
        critic: ValueCritic,
        optim: torch.optim.Optimizer,
        action_space: gym.spaces.Discrete, # Should be Discrete(num_nodes)
        # --- PPO Params ---
        eps_clip: float = 0.2, dual_clip: Optional[float] = None, value_clip: bool = False, advantage_normalization: bool = True,
        recompute_advantage: bool = False, vf_coef: float = 0.25, ent_coef: float = 0.1, gamma: float = 0.99, # Higher ENT_COEF?
        gae_lambda: float = 0.9, reward_normalization: bool = False, max_grad_norm: Optional[float] = None, **kwargs: Any, # Lower Lambda?
    ) -> None:
        # Check if diffusion model's action_dim matches the provided action_space size
        if diffusion_model.action_dim != action_space.n:
             raise ValueError(f"Diffusion model action_dim ({diffusion_model.action_dim}) "
                              f"must match action_space.n ({action_space.n}) for Deploy-Only actions.")

        super().__init__( actor=diffusion_model.model, critic=critic, optim=optim, dist_fn=Categorical, action_space=action_space, discount_factor=gamma, gae_lambda=gae_lambda, vf_coef=vf_coef, ent_coef=ent_coef, reward_normalization=reward_normalization, **kwargs )
        self.diffusion_model = diffusion_model; self._eps_clip = eps_clip; self._dual_clip = dual_clip; self._value_clip = value_clip; self._norm_adv = advantage_normalization; self._recompute_adv = recompute_advantage; self._lambda = gae_lambda; self._gamma = gamma; self._grad_norm = max_grad_norm; self._device = next(self.critic.parameters()).device; print(f"SFCDiffusionPPOPolicy (Deploy-Only) initialized on device: {self._device}")
        print(f"  Action Space Size: {action_space.n}")
        print(f"  Gamma: {self._gamma}, Lambda: {self._lambda}, EpsClip: {self._eps_clip}, EntCoef: {self._weight_ent}")


    def forward(
            self, batch: Batch, state: Optional[Union[dict, Batch, np.ndarray]] = None, input: str = "obs",
    ) -> Batch:
        obs_flat = to_torch(batch[input], device=self._device, dtype=torch.float32)

        raw_logits = self.diffusion_model(obs_flat) # Output dim must match action_space.n

        nan_inf_detected = False
        if torch.isnan(raw_logits).any() or torch.isinf(raw_logits).any():
            nan_inf_detected = True
            # if not hasattr(self, '_nan_warning_printed') or random.random() < 0.01: print(f"!!! WARNING: NaN/Inf detected in raw_logits. Replacing. !!!"); self._nan_warning_printed = True
            raw_logits = torch.nan_to_num(raw_logits, nan=-1e8, posinf=1e8, neginf=-1e8)

        # Sanity check
        if torch.isnan(raw_logits).any() or torch.isinf(raw_logits).any(): print(f"!!! CRITICAL ERROR: NaN/Inf DETECTED in raw_logits before Categorical! Origin NaN was {nan_inf_detected} !!!"); raw_logits = torch.nan_to_num(raw_logits, nan=-1e8, posinf=1e8, neginf=-1e8)

        dist = Categorical(logits=raw_logits) # Use raw logits
        act = dist.sample()

        return Batch(logits=raw_logits, act=act, state=state, dist=dist) # REMOVED mask=mask

    def process_fn(
        self, batch: Batch, buffer: ReplayBuffer, indices: np.ndarray
    ) -> Batch:
        if self._recompute_adv: self._buffer, self._indices = buffer, indices
        batch = self._compute_returns(batch, buffer, indices)
        batch.act = to_torch_as(batch.act, batch.v_s)
        with torch.no_grad():
             try:
                 batch_for_logp = Batch(obs=batch.obs) # Only need obs

                 dist_old = self(batch_for_logp).dist # self() uses raw_logits

                 current_act_batch_size = batch.act.shape[0]; dist_batch_size = dist_old.batch_shape[0]
                 if dist_batch_size != current_act_batch_size: print(f"Warning process_fn: Mismatch dist batch size ({dist_batch_size}) vs action ({current_act_batch_size}). Using min size."); min_size = min(dist_batch_size, current_act_batch_size); batch.logp_old = dist_old.log_prob(batch.act[:min_size]);
                 else: batch.logp_old = dist_old.log_prob(batch.act)
                 # Handle potential size mismatch if needed (e.g., padding logp_old)
                 if dist_batch_size < current_act_batch_size:
                      padding = torch.zeros(current_act_batch_size - dist_batch_size, *batch.logp_old.shape[1:], dtype=batch.logp_old.dtype, device=batch.logp_old.device); batch.logp_old = torch.cat([batch.logp_old, padding], dim=0)


             except Exception as e: print(f"Error calculating logp_old: {e}. Setting to zero."); batch.logp_old = torch.zeros_like(batch.act, dtype=torch.float32)
        return batch

    def _compute_returns( self, batch: Batch, buffer: ReplayBuffer, indices: np.ndarray ) -> Batch:
        v_s, v_s_ = [], [];
        with torch.no_grad():
            if "obs" not in batch or "obs_next" not in batch: raise ValueError("Batch must contain 'obs' and 'obs_next'")
            for minibatch in batch.split(self._batch, shuffle=False, merge_last=True):
                obs_flat = to_torch(minibatch.obs, device=self._device, dtype=torch.float32); obs_next_flat = to_torch(minibatch.obs_next, device=self._device, dtype=torch.float32)
                v_s.append(self.critic(obs_flat)); v_s_.append(self.critic(obs_next_flat))
        batch.v_s = torch.cat(v_s, dim=0).flatten(); v_s_ = torch.cat(v_s_, dim=0).flatten()
        batch.rew = to_torch_as(batch.rew, batch.v_s); batch.done = to_torch_as(batch.done, batch.v_s)
        adv = torch.zeros_like(batch.rew); gae = 0.0
        for i in reversed(range(len(batch.rew))): delta = batch.rew[i] + self._gamma * v_s_[i] * (1 - batch.done[i]) - batch.v_s[i]; gae = delta + self._gamma * self._lambda * (1 - batch.done[i]) * gae; adv[i] = gae
        batch.adv = adv; batch.returns = batch.adv + batch.v_s
        if self._norm_adv: mean, std = batch.adv.mean(), batch.adv.std(); batch.adv = (batch.adv - mean) / (std + self._eps)
        return batch

    def learn( self, batch: Batch, batch_size: Optional[int], repeat: int, **kwargs: Any ) -> Dict[str, List[float]]:
        if batch_size is None: batch_size = len(batch)
        losses, clip_losses, vf_losses, ent_losses = [], [], [], []
        for step in range(repeat):
             if self._recompute_adv and step > 0: batch = self.process_fn(batch, self._buffer, self._indices)
             for minibatch in batch.split(batch_size, merge_last=True):
                  if not all(hasattr(minibatch, k) for k in ['obs', 'act', 'logp_old', 'adv', 'returns']): raise ValueError(f"Minibatch missing required keys for learn. Has: {minibatch.keys()}")
                  curr_dist = self(minibatch).dist; logp_new = curr_dist.log_prob(minibatch.act)
                  ratio = (logp_new - minibatch.logp_old).exp().float().view(-1, 1); adv = minibatch.adv.view(-1, 1)
                  adv_clamp_val = 50.0; adv_before_clamp_mean = adv.mean().item(); adv = torch.clamp(adv, -adv_clamp_val, adv_clamp_val); adv_after_clamp_mean = adv.mean().item()
                  # if step == 0 and repeat > 1 : print(f"Debug learn: Adv Clamped from Mean={adv_before_clamp_mean:.3f} to Mean={adv_after_clamp_mean:.3f} | Ratio Mean={ratio.mean().item():.3f}")
                  if torch.isnan(adv).any() or torch.isinf(adv).any(): print(f"!!! ERROR: NaN/Inf in Advantage values AFTER CLAMPING !!!")
                  if torch.isnan(ratio).any() or torch.isinf(ratio).any(): print(f"!!! ERROR: NaN/Inf in Ratio values !!!")
                  surr1 = ratio * adv; surr2 = torch.clamp(ratio, 1.0 - self._eps_clip, 1.0 + self._eps_clip) * adv
                  if self._dual_clip: clip_loss = -torch.max(torch.min(surr1, surr2), self._dual_clip * adv).mean()
                  else: clip_loss = -torch.min(surr1, surr2).mean()
                  if torch.isnan(clip_loss).any() or torch.isinf(clip_loss).any(): print(f"!!! ERROR: NaN/Inf detected in clip_loss !!!"); continue
                  obs_flat = to_torch(minibatch.obs, device=self._device, dtype=torch.float32); value_pred = self.critic(obs_flat).flatten(); target_return = minibatch.returns
                  if self._value_clip: v_clip = minibatch.v_s + (value_pred - minibatch.v_s).clamp(-self._eps_clip, self._eps_clip); vf1 = (target_return - value_pred).pow(2); vf2 = (target_return - v_clip).pow(2); vf_loss = torch.max(vf1, vf2).mean()
                  else: vf_loss = (target_return - value_pred).pow(2).mean()
                  if torch.isnan(vf_loss).any() or torch.isinf(vf_loss).any(): print(f"!!! ERROR: NaN/Inf detected in vf_loss !!!"); continue
                  ent_loss = curr_dist.entropy().mean()
                  if torch.isnan(ent_loss).any(): print(f"!!! ERROR: NaN detected in ent_loss !!!"); ent_loss = torch.tensor(0.0)
                  loss = clip_loss + self._weight_vf * vf_loss - self._weight_ent * ent_loss
                  if torch.isnan(loss).any() or torch.isinf(loss).any(): print(f"!!! ERROR: NaN/Inf detected in total loss !!!"); continue
                  self.optim.zero_grad(); loss.backward()
                  if self._grad_norm: nn.utils.clip_grad_norm_(self.parameters(), max_norm=self._grad_norm)
                  self.optim.step()
                  clip_losses.append(clip_loss.item()); vf_losses.append(vf_loss.item()); ent_losses.append(ent_loss.item()); losses.append(loss.item())
        return {"loss": losses, "loss/clip": clip_losses, "loss/vf": vf_losses, "loss/ent": ent_losses}

    def parameters(self): yield from self.critic.parameters(); yield from self.diffusion_model.model.parameters()