from config import VexConfig
from eval.evaluator import Evaluator
from model.mlp import MLP
from env.env import VexEnv

import torch 

def eval_workers(
    worker_id: int,
    evaluator: Evaluator,
    config: VexConfig,
):
    torch.set_num_threads(1)
    env = VexEnv(config)
    model_1 = MLP(config)
    model_2 = MLP(config)
    model_1.eval()
    model_2.eval()
    print(f"Eval Worker {worker_id} started.", flush=True)
    while True:
        pair = evaluator.get_next_matchup()
        if pair is None:
            # shutdown signal, all matchups are done.
            print(f"Eval Worker {worker_id} received shutdown signal. All matchups are done. Exiting.", flush=True)
            break
        idx1, idx2, param1, param2 = pair    
        torch.nn.utils.vector_to_parameters(param1, model_1.parameters())
        torch.nn.utils.vector_to_parameters(param2, model_2.parameters())
        env_out = env.reset()
        done, legal_actions, observations, rewards, timestep = \
            env_out['done'], env_out['legal_actions'], env_out['observations'], env_out['rewards'], env_out['timestep']
        while not done:
            with torch.no_grad():
                out1 = model_1(
                    observations['robot_red']['core_obs'].view(1, -1), # (1, core_obs_dim)
                    observations['robot_red']['ball_obs'].view(1, config.n_balls, -1), # (1, n_balls, ball_obs_dim)
                    legal_actions['robot_red'].view(1, -1), # (1, n_primary_actions)
                    inference=True
                )
                out2 = model_2(
                    observations['robot_blue']['core_obs'].view(1, -1), # (1, core_obs_dim)
                    observations['robot_blue']['ball_obs'].view(1, config.n_balls, -1), # (1, n_balls, ball_obs_dim)
                    legal_actions['robot_blue'].view(1, -1), # (1, n_primary_actions)
                    inference=True
                )
            env_out = env.step(out1['action'], out2['action'])
            done, legal_actions, observations, rewards, timestep = \
                env_out['done'], env_out['legal_actions'], env_out['observations'], env_out['rewards'], env_out['timestep']
        
        red_won = rewards['robot_red'] > rewards['robot_blue']
        blue_won = rewards['robot_blue'] > rewards['robot_red']        
    