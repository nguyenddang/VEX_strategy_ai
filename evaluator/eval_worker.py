from config import VexConfig
from model.mlp import MLP
from env.env import VexEnv

import torch 
import time
import random

from evaluator.evaluator import TrainEvaluator
def eval_workers(
    worker_id: int,
    evaluator: TrainEvaluator,
    config: VexConfig,
):
    torch.set_num_threads(1)
    time.sleep(worker_id * 0.01)
    env = VexEnv(config)
    test_model = MLP(config)
    ref_model = MLP(config)
    test_model.eval()
    ref_model.eval()
    print(f"Eval Worker {worker_id} started.", flush=True)
    while True:
        pair = evaluator.get_next_matchup()
        if pair is None:
            time.sleep(5)
            continue
        k1 = random.choice(['robot_red', 'robot_blue'])
        k2 = 'robot_blue' if k1 == 'robot_red' else 'robot_red'
        test_version, ref_idx = pair 
        with evaluator.lock:
            torch.nn.utils.vector_to_parameters(evaluator.test_param, test_model.parameters())
            torch.nn.utils.vector_to_parameters(evaluator.ref_params[ref_idx], ref_model.parameters())
        env_out = env.reset()
        done, legal_actions, observations, rewards, timestep = \
            env_out['done'], env_out['legal_actions'], env_out['observations'], env_out['rewards'], env_out['timestep']
        while not done:
            with torch.no_grad():
                out1 = test_model(
                    observations[k1]['core_obs'].view(1, -1), # (1, core_obs_dim)
                    observations[k1]['ball_obs'].view(1, config.n_balls, -1), # (1, n_balls, ball_obs_dim)
                    legal_actions[k1].view(1, -1), # (1, n_primary_actions)
                    inference=True
                )
                out2 = ref_model(
                    observations[k2]['core_obs'].view(1, -1), # (1, core_obs_dim)
                    observations[k2]['ball_obs'].view(1, config.n_balls, -1), # (1, n_balls, ball_obs_dim)
                    legal_actions[k2].view(1, -1), # (1, n_primary_actions)
                    inference=True
                )
            actions = {
                k1: out1['actions'][0].tolist(), # [discrete, x, y, theta]
                k2: out2['actions'][0].tolist(),
            }
            env_out = env.step(actions)
            done, legal_actions, observations, rewards, timestep = \
                env_out['done'], env_out['legal_actions'], env_out['observations'], env_out['rewards'], env_out['timestep']
        
        test_won = env_out['score'][k1] > env_out['score'][k2]
        ref_won = env_out['score'][k2] > env_out['score'][k1]   
        evaluator.update_trueskill(test_version, ref_idx, test_won, ref_won)
    