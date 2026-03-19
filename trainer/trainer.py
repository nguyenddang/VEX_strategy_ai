from config import VexConfig
import wandb

from model.mlp import MLP

from trainer.shared import SharedBuffer, SharedLeague
from trainer.worker import worker_decentralized_fn

from evaluator.evaluator import TrainEvaluator
from evaluator.eval_worker import eval_workers

from torch.distributions import Categorical
import torch.nn.functional as F
import torch.multiprocessing as mp
import torch

import time 

class Trainer:
    def __init__(self, config: VexConfig):
        self.config = config

        # train configs
        self.n_workers = config.n_workers
        self.update_league = config.update_league

        # train iteration config
        self.iteration = 0
        self.steps_per_iteration = config.steps_per_iteration
        self.train_device = config.train_device
        self.core_obs_dim = config.core_obs_dim
        self.ball_obs_dim = config.ball_obs_dim
        self.n_balls = config.n_balls
        self.n_primary_actions = config.n_primary_actions
        self.train_episodes = config.train_episodes
        self.mini_train_episodes = config.mini_train_episodes
        self.max_actions = config.max_actions
        self.total_timesteps = config.total_timesteps
        self.grad_accumulation_steps = int(self.train_episodes // config.mini_train_episodes)
        # gae config
        self.gamma = config.gamma
        self.lam = config.lam
        # loss config
        self.value_epsilon = config.value_epsilon
        self.policy_epsilon = config.policy_epsilon
        self.entropy_coef = config.entropy_coef
        self.value_coef = config.value_coef

        # init model, shared buffer, league, and inference server.
        self.learner = MLP(config)
        self.learner.train()
        self.buffer = SharedBuffer(config)
        self.league = SharedLeague(config)
        self.evaluator = TrainEvaluator(config, self.league)

        # init optimizer
        self.optimizer = torch.optim.AdamW(self.learner.parameters(), lr=config.lr)
        self.learner.to(self.train_device)
        if config.resume_training:
            print(f"Resuming training from checkpoint: {config.load_ckpt_path}", flush=True)
            ckpt = torch.load(config.load_ckpt_path)
            self.load_state_dict(ckpt)
        else:
            print("Starting training from scratch.", flush=True)
            self.league.update_latest_snapshot(self.learner)
            self.league.update_learner_param(self.learner)
        # misc
        self.last_produced = 0

        # start processes
        self.processes = []

        for worker_id in range(self.n_workers):
            p = mp.Process(target=worker_decentralized_fn, args=(worker_id, self.buffer, self.league, config))
            p.start()
            self.processes.append(p)
            
        for eval_worker_id in range(config.n_eval_workers):
            p = mp.Process(target=eval_workers, args=(eval_worker_id, self.evaluator, config))
            p.start()
            self.processes.append(p)
        
        if config.compile:
            print("Compiling model...", flush=True)
            self.learner = torch.compile(self.learner)
        print(f"Gradient accumulation steps: {self.grad_accumulation_steps}", flush=True)

        # logging
        self.log_wandb = config.log_wandb
        if self.log_wandb:
            wandb.init(project=config.wandb_project, name=config.wandb_run_name, config=config.__dict__)

        # saving ckpt configs
        self.save_ckpt_path = config.save_ckpt_path
        self.n_save_learner_ckpts = config.n_save_learner_ckpts
        self.n_save_all_ckpts = config.n_save_all_ckpts


    def _get_advantage(self, rewards: torch.Tensor, values: torch.Tensor):
        """
        reward: batch_size, total_timesteps
        values: batch_size, total_timesteps + 1
        """
        advantages = torch.zeros_like(rewards).float()
        gae = torch.zeros((rewards.shape[0],), device=self.train_device)
        for t in reversed(range(rewards.shape[1])):
            delta = rewards[:, t] + self.gamma * values[:, t + 1] - values[:, t]
            gae = delta + self.gamma * self.lam * gae
            advantages[:, t] = gae
        return advantages
    
    def _get_loss(self, values, ratios, advantages, returns,
                  p_dist: Categorical, x_dist: Categorical, y_dist: Categorical, theta_dist: Categorical, move_masks: torch.Tensor):
        value_loss = F.mse_loss(values, returns, reduction='mean')
        policy_loss = -torch.min(
            ratios * advantages,
            torch.clamp(ratios, 1.0 - self.policy_epsilon, 1.0 + self.policy_epsilon) * advantages).mean()
        entropy_bonus = (p_dist.entropy() + move_masks * (x_dist.entropy() + y_dist.entropy() + theta_dist.entropy())).mean()
        total_loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy_bonus
        return total_loss, policy_loss, value_loss, entropy_bonus

    def _train_iteration(self):
        accum_learner_versions = 0.0
        iteration_batch = {
            "policy_loss": 0.0,
            "value_loss": 0.0,
            "entropy_bonus": 0.0,
            "grad_norm": 0.0,
            "adv_mean": 0.0,
            "adv_std": 0.0,
            "learner_score": 0.0,
            "opp_score": 0.0,
            'action_percentage': torch.zeros((self.n_primary_actions,), dtype=torch.float32),
        }
        denom = self.steps_per_iteration * self.grad_accumulation_steps
        for step in range(self.steps_per_iteration):
            for g_step in range(self.grad_accumulation_steps):
                time.sleep(0.05) # wait for buffer to be filled
                batch = self.buffer.pull_from_buffer()
                core_obs = batch['core_obs'].pin_memory().to(self.train_device, non_blocking=True).view(-1, self.core_obs_dim) # B * total_timesteps, core_obs_dim
                ball_obs = batch['ball_obs'].pin_memory().to(self.train_device, non_blocking=True).view(-1, self.n_balls, self.ball_obs_dim) # B * total_timesteps, n_balls, ball_obs_dim
                legal_masks = batch['legal_masks'].pin_memory().to(self.train_device, non_blocking=True).view(-1, self.n_primary_actions) # B * total_timesteps, n_primary_actions
                actions = batch['actions'].pin_memory().to(self.train_device, non_blocking=True).view(-1, 4) # B * total_timesteps, 4
                rewards = batch['rewards'].pin_memory().to(self.train_device, non_blocking=True) # B, total_timesteps
                values = batch['values'].pin_memory().to(self.train_device, non_blocking=True) # B, total_timesteps + 1
                move_masks = batch['move_masks'].pin_memory().to(self.train_device, non_blocking=True).view(-1) # B * 2 * total_timesteps
                log_probs = batch['log_probs'].pin_memory().to(self.train_device, non_blocking=True).view(-1) # B * 2 * total_timesteps
                accum_learner_versions += batch['learner_versions'].mean().item()
                learner_score = batch['learner_score'].mean().item()
                opp_score = batch['opp_score'].mean().item()

                with torch.no_grad():
                    advantages = self._get_advantage(rewards, values) # B, total_timesteps
                    returns = (advantages + values[:, :-1]).view(-1) # B * total_timesteps
                    adv_mean = advantages.mean()
                    adv_std = advantages.std() + 1e-8
                    advantages = (advantages.view(-1) - adv_mean) / adv_std # B * total_timesteps
                with torch.autocast(device_type=self.train_device, dtype=torch.bfloat16, enabled=(self.train_device == 'cuda:0')):
                    outputs = self.learner(core_obs, ball_obs, legal_masks)
                    p_dist = Categorical(logits=outputs["primary_action_logits"])
                    x_dist = Categorical(logits=outputs["x_bin_logits"])
                    y_dist = Categorical(logits=outputs["y_bin_logits"])
                    theta_dist = Categorical(logits=outputs["theta_bin_logits"])
                    p_prob = p_dist.log_prob(actions[:, 0])
                    x_prob = x_dist.log_prob(actions[:, 1])
                    y_prob = y_dist.log_prob(actions[:, 2])
                    theta_prob = theta_dist.log_prob(actions[:, 3])
                    new_log_probs = p_prob + move_masks * (x_prob + y_prob + theta_prob)
                    ratios = torch.exp(new_log_probs - log_probs)
                    new_values = outputs["value_logits"]

                    loss, policy_loss, value_loss, entropy_bonus = self._get_loss(
                        new_values, ratios, advantages, returns,
                        p_dist, x_dist, y_dist, theta_dist, move_masks
                    )
                    loss /= self.grad_accumulation_steps
                loss.backward()
                iteration_batch['policy_loss'] += policy_loss.item() / denom
                iteration_batch['value_loss'] += value_loss.item() / denom
                iteration_batch['entropy_bonus'] += entropy_bonus.item() / denom
                iteration_batch['adv_mean'] += adv_mean.item() / denom
                iteration_batch['adv_std'] += adv_std.item() / denom
                iteration_batch['learner_score'] += learner_score / denom
                iteration_batch['opp_score'] += opp_score / denom
                iteration_batch['action_percentage'] += torch.bincount(actions[:, 0].cpu(), minlength=self.n_primary_actions).float() / (actions.shape[0] * denom)
            norm = torch.nn.utils.clip_grad_norm_(self.learner.parameters(), max_norm=1.0)
            iteration_batch['grad_norm'] += norm.item() / self.steps_per_iteration
            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)
            print(f"Step {step}: Loss {(self.grad_accumulation_steps * loss).item():.4f}, \
Policy Loss {policy_loss.item():.4f}, \
Value Loss {value_loss.item():.4f}, \
Entropy Bonus {entropy_bonus.item():.4f}, \
Grad Norm {norm:.4f}, \
Adv Mean {adv_mean.item():.4f}, \
Adv Std {adv_std.item():.4f}, \
Learner Score {learner_score:.4f} \
Opponent Score {opp_score:.4f}", flush=True)
        return accum_learner_versions / (self.steps_per_iteration * self.grad_accumulation_steps), iteration_batch


    def train(self):
        try:
            while True:
                print(f"Starting iteration {self.iteration}", flush=True)
                start_time = time.time()
                M = float(self.iteration)
                N, iteration_batch = self._train_iteration()

                self.league.update_learner_param(self.learner)
                self.evaluator.update_learner_param(self.learner, self.iteration + 1)
                if self.iteration % self.update_league == 0 and self.iteration > 0:
                    print(f"Updating league at iteration {self.iteration}", flush=True)
                    self.league.update_latest_snapshot(self.learner)
                # torch.cuda.synchronize()
                end_time = time.time()
                
                dt = end_time - start_time
                sample_reuse_batch = self.get_sample_reuse(dt, self.iteration)
                print(f"[TRAINER] Iteration: {self.iteration}, Staleness: {(M-N):.2f}, Optimizing Version: {M:.2f}, Average Learner Version in Batch: {N:.2f}", flush=True)
                with self.evaluator.lock:
                    print(f"[TRAINER] Current TrueSkill of Learner: {self.evaluator.test_mu.value:.2f} ± {self.evaluator.test_sigma.value:.2f}, Evaluated Version: {self.evaluator.test_version.value} Games Played: {self.evaluator.test_n.value}", flush=True)
                # logging
                if self.log_wandb:
                    iteration_batch['move%'] = iteration_batch['action_percentage'][0].item()
                    iteration_batch['pick_loader%'] = iteration_batch['action_percentage'][1].item()
                    iteration_batch['pick_ground%'] = iteration_batch['action_percentage'][2].item()
                    iteration_batch['score%'] = iteration_batch['action_percentage'][3].item()
                    iteration_batch['block%'] = iteration_batch['action_percentage'][4].item()
                    with self.evaluator.lock:
                        iteration_batch['trueskill'] = self.evaluator.test_mu.value
                        iteration_batch['trueskill_sigma'] = self.evaluator.test_sigma.value
                        iteration_batch['trueskill_v'] = self.evaluator.test_version.value
                    del iteration_batch['action_percentage']
                    wandb.log(iteration_batch | sample_reuse_batch | {"staleness": (M-N), "optimizing_version": M, "average_learner_version_in_batch": N}, step=self.iteration)


                self.iteration += 1
                # saving ckpts
                if self.iteration > 0 and (self.iteration % self.n_save_learner_ckpts) == 0:
                    learner_ckpt_path = f"{self.save_ckpt_path}/learner_{self.iteration}.pt"
                    torch.save(self.state_dict_learner(self.iteration), learner_ckpt_path)
                    print(f"Saved learner checkpoint at iteration {self.iteration} to {learner_ckpt_path}", flush=True)
                if self.iteration > 0 and (self.iteration % self.n_save_all_ckpts) == 0:
                    all_ckpt_path = f"{self.save_ckpt_path}/all.pt"
                    torch.save(self.state_dict(self.iteration), all_ckpt_path)
                    print(f"Saved full checkpoint at iteration {self.iteration} to {all_ckpt_path}", flush=True)
                
                

        except KeyboardInterrupt:
            print("KeyboardInterrupt received. Stopping training loop...", flush=True)
        except Exception as e:
            print(f"Exception in training loop: {e}", flush=True)
        finally:            
            print("Terminating worker processes...", flush=True)
            for p in self.processes:
                if p.is_alive():
                    p.terminate()
            for p in self.processes:
                p.join()
            print("All worker processes terminated.", flush=True)
    
    def state_dict_learner(self, iteration):
        return {
            'model': self.learner.state_dict(),
            'optim': self.optimizer.state_dict(),
            'iteration': iteration,
            'config': self.config.__dict__
        }
    
    def state_dict(self, iteration):
        learner = self.state_dict_learner(iteration)
        league = self.league.state_dict()
        evaluator = self.evaluator.state_dict()
        return learner | league | evaluator
    
    def load_state_dict_learner(self, state_dict):
        unwanted_prefix = '_orig_mod.'
        for k,v in list(state_dict['model'].items()):
            if k.startswith(unwanted_prefix):
                state_dict['model'][k[len(unwanted_prefix):]] = state_dict['model'].pop(k)
        self.learner.load_state_dict(state_dict['model'])
        self.optimizer.load_state_dict(state_dict['optim'])
        self.iteration = state_dict['iteration']
    
    def load_state_dict(self, state_dict):
        self.load_state_dict_learner(state_dict)
        self.league.load_state_dict(state_dict)
        self.evaluator.load_state_dict(state_dict)

    def get_sample_reuse(self, dt, iteration):
        with self.buffer.sample_produced.get_lock():
            produced = self.buffer.sample_produced.value
        sample_trained_per_sec = self.train_episodes * self.max_actions * 2 * self.steps_per_iteration / dt
        sample_produced_per_sec = (produced - self.last_produced) / dt
        self.last_produced = produced
        if sample_produced_per_sec == 0:
            return float('inf')
        sample_reuse = sample_trained_per_sec / sample_produced_per_sec
        print(f"[TRAINER] Iteration: {iteration}, Sample Trained/sec: {sample_trained_per_sec:.2f}, Sample Produced/sec: {sample_produced_per_sec:.2f}, Sample Reuse: {sample_reuse:.2f}", flush=True)
        return {"sample_reuse": sample_reuse, "sample_trained_per_sec": sample_trained_per_sec, "sample_produced_per_sec": sample_produced_per_sec}
        