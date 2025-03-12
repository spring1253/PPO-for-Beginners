import gymnasium as gym
import time
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.distributions import MultivariateNormal

class PPO:
    def __init__(self, policy_class, env, **hyperparameters):
        assert isinstance(env.observation_space, gym.spaces.Box)
        assert isinstance(env.action_space, gym.spaces.Box)
        
        self._init_hyperparameters(hyperparameters)
        self.env = env
        self.obs_dim = env.observation_space.shape[0]
        self.act_dim = env.action_space.shape[0]

        self.actor = policy_class(self.obs_dim, self.act_dim)
        self.critic = policy_class(self.obs_dim, 1)

        self.actor_optim = Adam(self.actor.parameters(), lr=self.lr)
        self.critic_optim = Adam(self.critic.parameters(), lr=self.lr)

        self.cov_var = torch.full(size=(self.act_dim,), fill_value=0.5)
        self.cov_mat = torch.diag(self.cov_var)

        self.logger = {'t_so_far': 0, 'i_so_far': 0, 'actor_losses': []}

    def learn(self, total_timesteps, epsilon):
        t_so_far, i_so_far = 0, 0
        start_time = time.time()
        while t_so_far < total_timesteps:
            batch_obs, batch_acts, batch_log_probs, batch_rtgs, batch_lens = self.rollout()
            t_so_far += np.sum(batch_lens)
            i_so_far += 1
            self.logger['t_so_far'] = t_so_far
            self.logger['i_so_far'] = i_so_far

            self.finite_difference_update(batch_obs, batch_acts, batch_log_probs, batch_rtgs, epsilon)

            if i_so_far % self.save_freq == 0:
                torch.save(self.actor.state_dict(), f'./models/ppo_actor_ver2_{total_timesteps//2200}_{epsilon}.pth')
                torch.save(self.critic.state_dict(), f'./models/ppo_critic_ver2_{total_timesteps//2200}_{epsilon}.pth')
            
            self._log_summary()
        end_time = time.time()
        with open(f'./models/ppo_actor_ver2_{total_timesteps//2200}_{epsilon}.txt', 'w') as file:
            file.write(f"{end_time - start_time}")

    def finite_difference_update(self, batch_obs, batch_acts, batch_log_probs, batch_rtgs, epsilon):
        params = list(self.actor.parameters())
        gradients = [torch.zeros_like(p) for p in params]
        
        for i, param in enumerate(params):
            perturb = torch.zeros_like(param)
            for index in range(param.numel()):
                perturb.view(-1)[index] = epsilon
                param.data += perturb
                loss_plus = self.surrogate_loss(batch_obs, batch_acts, batch_log_probs, batch_rtgs)
                param.data -= 2 * perturb
                loss_minus = self.surrogate_loss(batch_obs, batch_acts, batch_log_probs, batch_rtgs)
                param.data += perturb
                gradients[i].view(-1)[index] = (loss_plus - loss_minus) / (2 * epsilon)
                perturb.view(-1)[index] = 0
        
        with torch.no_grad():
            for param, grad in zip(params, gradients):
                param -= self.lr * grad
    
    def surrogate_loss(self, batch_obs, batch_acts, batch_log_probs, batch_rtgs):
        V, curr_log_probs = self.evaluate(batch_obs, batch_acts)
        advantages = batch_rtgs - V.detach()
        ratios = torch.exp(curr_log_probs - batch_log_probs)
        surr1 = ratios * advantages
        surr2 = torch.clamp(ratios, 1 - self.clip, 1 + self.clip) * advantages
        return -torch.min(surr1, surr2).mean()

    def evaluate(self, batch_obs, batch_acts):
        V = self.critic(batch_obs).squeeze()
        mean = self.actor(batch_obs)
        dist = MultivariateNormal(mean, self.cov_mat)
        log_probs = dist.log_prob(batch_acts)
        return V, log_probs

    def rollout(self):
        batch_obs, batch_acts, batch_log_probs, batch_rews, batch_rtgs, batch_lens = [], [], [], [], [], []
        t = 0
        while t < self.timesteps_per_batch:
            ep_rews = []
            obs, _ = self.env.reset()
            done = False
            for ep_t in range(self.max_timesteps_per_episode):
                action, log_prob = self.get_action(obs)
                obs, rew, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                ep_rews.append(rew)
                batch_obs.append(obs)
                batch_acts.append(action)
                batch_log_probs.append(log_prob)
                t += 1
                if done:
                    break
            batch_lens.append(ep_t + 1)
            batch_rews.append(ep_rews)
        batch_obs = torch.tensor(np.array(batch_obs), dtype=torch.float)
        batch_acts = torch.tensor(np.array(batch_acts), dtype=torch.float)
        batch_log_probs = torch.tensor(batch_log_probs, dtype=torch.float)
        batch_rtgs = self.compute_rtgs(batch_rews)
        return batch_obs, batch_acts, batch_log_probs, batch_rtgs, batch_lens
    
    def compute_rtgs(self, batch_rews):
        batch_rtgs = []
        for ep_rews in reversed(batch_rews):
            discounted_reward = 0
            for rew in reversed(ep_rews):
                discounted_reward = rew + discounted_reward * self.gamma
                batch_rtgs.insert(0, discounted_reward)
        return torch.tensor(batch_rtgs, dtype=torch.float)
    
    def get_action(self, obs):
        mean = self.actor(obs)
        dist = MultivariateNormal(mean, self.cov_mat)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action.detach().numpy(), log_prob.detach()
    
    def _log_summary(self):
        print(f"Iteration {self.logger['i_so_far']}: ")
        print(f"    Timesteps So Far: {self.logger['t_so_far']}")
        if self.logger['actor_losses']:
            print(f"    Average Actor Loss: {np.mean(self.logger['actor_losses'][-10:])}")
        print("----------------------------------")
    
    def _init_hyperparameters(self, hyperparameters):
        self.timesteps_per_batch = 4800
        self.max_timesteps_per_episode = 1600
        self.lr = 0.005
        self.gamma = 0.95
        self.clip = 0.2
        self.save_freq = 10
        for param, val in hyperparameters.items():
            setattr(self, param, val)
