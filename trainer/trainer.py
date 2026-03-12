from config import VexConfig

from model.model import AgentMLP

from trainer.shared import SharedBuffer, SharedLeague
from trainer.inference import InferenceServer
from trainer.worker import worker_fn

from torch.distributions import Categorical
import torch.nn.functional as F
import torch.multiprocessing as mp
import torch

import time 

class Trainer:
    def __init__(self, config: VexConfig):
        # train configs
        self.n_workers = config.n_workers
        self.update_league = config.update_league

        # train iteration config
        self.steps_per_iteration = config.steps_per_iteration
        self.train_device = config.train_device
        self.core_obs_dim = config.core_obs_dim
        self.ball_obs_dim = config.ball_obs_dim
        self.n_balls = config.n_balls
        self.n_primary_actions = config.n_primary_actions
        self.train_batch_size = config.train_batch_size

        # gae config
        self.gamma = config.gamma
        self.lam = config.lam

        # loss config
        self.value_epsilon = config.value_epsilon
        self.policy_epsilon = config.policy_epsilon
        self.entropy_coef = config.entropy_coef
        self.value_coef = config.value_coef

        # init model, shared buffer, league, and inference server.
        self.learner = AgentMLP(config)
        self.buffer = SharedBuffer(config)
        self.league = SharedLeague(config)
        self.inference_server = InferenceServer(self.buffer, self.league, config)
    
        self.league.update_learner_param(self.learner)
        self.league.update_latest_snapshot(self.learner)

        # init optimizer
        self.optimizer = torch.optim.Adam(self.learner.parameters(), lr=config.lr)
        
        # misc
        self.last_produced = 0

        # start processes
        self.processes = []
        server_proc = mp.Process(target=self.inference_server.run, args=())
        server_proc.start()
        self.processes.append(server_proc)

        for worker_id in range(self.n_workers):
            p = mp.Process(target=worker_fn, args=(worker_id, self.buffer, self.league, config))
            p.start()
            self.processes.append(p)
        
        self.learner.to(self.train_device)


    def _get_advantage(self, rewards: torch.Tensor, values: torch.Tensor):
        """
        reward: batch_size, 2, chunk_size
        values: batch_size, 2, chunk_size + 1
        """
        advantages = torch.zeros_like(rewards).float()
        gae = torch.zeros((rewards.shape[0], rewards.shape[1]), device=self.train_device)
        for t in reversed(range(rewards.shape[2])):
            delta = rewards[:, :, t] + self.gamma * values[:, :, t + 1] - values[:, :, t]
            gae = delta + self.gamma * self.lam * gae
            advantages[:, :, t] = gae
        return advantages
    
    def _get_loss(self, values, values_old, ratios, advantages, returns, train_mask,
                  p_dist: Categorical, x_dist: Categorical, y_dist: Categorical, theta_dist: Categorical, move_masks: torch.Tensor):
        """
        
        """
        value_loss = torch.max(
            F.mse_loss(values, returns, reduction='none'),
            F.mse_loss(
                torch.clamp(values, values_old - self.value_epsilon, values_old + self.value_epsilon),
                returns,
                reduction='none')).mean()
        ratios = ratios[train_mask]
        advantages = advantages[train_mask]
        policy_loss = -torch.min(
            ratios * advantages,
            torch.clamp(ratios, 1.0 - self.policy_epsilon, 1.0 + self.policy_epsilon) * advantages).mean()
        entropy_bonus = (p_dist.entropy() + move_masks * (x_dist.entropy() + y_dist.entropy() + theta_dist.entropy()))[train_mask].mean()
        total_loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy_bonus
        return total_loss, policy_loss, value_loss, entropy_bonus

    def _train_iteration(self):
        for step in range(self.steps_per_iteration):
            # DEBUG: Sleep for 1s then continue 
            # batch = self.buffer.pull_from_buffer()
            time.sleep(0.05)
            continue
            core_obs = batch['core_obs'].pin_memory().to(self.train_device, non_blocking=True).contiguous().view(-1, self.core_obs_dim) # B * 2 * chunk_size, core_obs_dim
            ball_obs = batch['ball_obs'].pin_memory().to(self.train_device, non_blocking=True).contiguous().view(-1, self.n_balls, self.ball_obs_dim) # B * 2 * chunk_size, n_balls, ball_obs_dim
            legal_masks = batch['legal_masks'].pin_memory().to(self.train_device, non_blocking=True).contiguous().view(-1, self.n_primary_actions) # B * 2 * chunk_size, n_primary_actions
            actions = batch['actions'].pin_memory().to(self.train_device, non_blocking=True).contiguous().view(-1, 4)
            rewards = batch['rewards'].pin_memory().to(self.train_device, non_blocking=True).contiguous()
            values = batch['values'].pin_memory().to(self.train_device, non_blocking=True)
            move_masks = batch['move_masks'].pin_memory().to(self.train_device, non_blocking=True).contiguous().view(-1) # B * 2 * chunk_size
            log_probs = batch['log_probs'].pin_memory().to(self.train_device, non_blocking=True).contiguous().view(-1) # B * 2 * chunk_size

            with torch.no_grad():
                advantages = self._get_advantage(rewards, values)
                returns = advantages + values[:, :, :-1]
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            with torch.autocast(device_type=self.train_device, dtype=torch.bfloat16, enabled=(self.train_device == 'cuda')):
                outputs = self.learner(core_obs, ball_obs, legal_masks)
                p_dist = Categorical(logits=outputs["primary_action_logits"])
                x_dist = Categorical(logits=outputs["x_bin_logits"])
                y_dist = Categorical(logits=outputs["y_bin_logits"])
                theta_dist = Categorical(logits=outputs["theta_bin_logits"])
                assert actions[:, 0].dtype == torch.long
                assert actions[:, 0].min() >= 0 and actions[:, 0].max() < 6
                p_prob = p_dist.log_prob(actions[:, 0])
                x_prob = x_dist.log_prob(actions[:, 1])
                y_prob = y_dist.log_prob(actions[:, 2])
                theta_prob = theta_dist.log_prob(actions[:, 3])

                new_log_probs = p_prob + move_masks * (x_prob + y_prob + theta_prob)
                ratios = torch.exp(new_log_probs - log_probs)

                new_values = outputs["value_logits"].squeeze(-1)
                values = values[:, :, :-1].contiguous().view(-1)

                train_mask = torch.arange(0, new_values.shape[0], 2, device=self.train_device)

                loss, policy_loss, value_loss, entropy_bonus = self._get_loss(
                    new_values, values, ratios, advantages.view(-1), returns.view(-1), train_mask,
                    p_dist, x_dist, y_dist, theta_dist, move_masks
                )
                
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(self.learner.parameters(), max_norm=1.0)
            self.optimizer.step()
            self.optimizer.zero_grad()
            print(f"Step {step}: Loss {loss.item():.4f}, Policy Loss {policy_loss.item():.4f}, Value Loss {value_loss.item():.4f}, Entropy Bonus {entropy_bonus.item():.4f}, Grad Norm {norm:.4f}")

    def train(self):
        iteration = 0
        while True:
            print(f"Starting iteration {iteration}")
            start_time = time.time()
            self._train_iteration()
            self.league.update_learner_param(self.learner)
            if iteration % self.update_league == 0 and iteration > 0:
                print(f"Updating league at iteration {iteration}")
                self.league.update_latest_snapshot(self.learner)
            iteration += 1
            # torch.cuda.synchronize()
            end_time = time.time()
            # Logging. 
            # long sample reuse = (samples per batch * batches per second) / (samples buffer intake per second)
            dt = end_time - start_time
            self.get_sample_reuse(dt, iteration)

    def get_sample_reuse(self, dt, iteration):
        with self.buffer.sample_read_lock:
            produced = self.buffer.sample_produced.value
        sample_trained_per_sec = self.train_batch_size * self.steps_per_iteration / dt
        sample_produced_per_sec = (produced - self.last_produced) / dt
        self.last_produced = produced
        if sample_produced_per_sec == 0:
            return float('inf')
        sample_reuse = sample_trained_per_sec / sample_produced_per_sec
        print(f"[TRAINER] Iteration: {iteration}, Sample Trained/sec: {sample_trained_per_sec:.2f}, Sample Produced/sec: {sample_produced_per_sec:.2f}, Sample Reuse: {sample_reuse:.2f}")
        