from pathlib import Path
from model.model import AgentMLP
import random
import sys
import time
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from env.env import VexEnv
from config import VexConfig

def run_demo(episodes: int = 1) -> None:
    config = VexConfig()
    config.render_mode = None
    env = VexEnv(config)
    learner = AgentMLP(config)
    learner_ckpt = torch.load("model_600.pt", map_location="cpu")
    learner.load_state_dict(learner_ckpt)
    learner.eval()

    try:
        for _ in range(episodes):
            out = env.reset()
            start_time = time.time()
            cum_reward = {'robot_red': 0.0, 'robot_blue': 0.0}
            while not out["done"]:
                action = {}
                legal_actions = out["legal_actions"]
                observations = out["observations"]

                action = {}
                for player in ["robot_red", "robot_blue"]:
                    legal_moves = legal_actions[player]
                    core_obs = observations[player]["core_obs"].view(1, -1)
                    ball_obs = observations[player]["ball_obs"].view(1, config.n_balls, -1)
                    legal_action_mask = legal_moves.view(1, -1)

                    outputs = learner(core_obs, ball_obs, legal_action_mask=legal_action_mask, inference=True)
        
                    action[player] = outputs["action"].squeeze(0).tolist()
                out = env.step(action)
                cum_reward['robot_red'] += out['reward']['robot_red']
                cum_reward['robot_blue'] += out['reward']['robot_blue']

            end_time = time.time()
            print(f"Episode(s) finished in {end_time - start_time:.2f} seconds.")
            print(f"Red score: {env.field.red_score}, Blue score: {env.field.blue_score}")
            print(f"Cumulative reward: {cum_reward}")

    finally:
        env.close()
if __name__ == "__main__":
    torch.set_num_threads(1)
    run_demo(episodes=5)
