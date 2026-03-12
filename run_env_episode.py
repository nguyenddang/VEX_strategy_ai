from pathlib import Path
import random
import sys
import time

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from env.env import VexEnv
from config import VexConfig
from model.model import AgentMLP
import torch 
config = VexConfig()
model = AgentMLP(config)
RANDOM_ACTION = False
def run_demo(episodes: int = 1) -> None:
    env = VexEnv(
        VexConfig(
            render_mode=None,
            engine_hz=60.0,
            inference_hz=5.0,
            max_duration_s=120.0,
            render_hz=30.0,
            realtime_render=False
        ),
    )
    
    try:
        for _ in range(episodes):
            out = env.reset()
            inference_time = 0
            start_time = time.time()
            cum_reward = {'robot_red': 0.0, 'robot_blue': 0.0}
            while not out["done"]:
                observations = out['observations']
                legal_actions = out['legal_actions']
                actions = {}
                if RANDOM_ACTION:
                    for robot_key in ['robot_red', 'robot_blue']:
                        action_idx = random.choice([i for i in range(6) if legal_actions[robot_key][i] == 1])
                        actions[robot_key] = [action_idx, random.randint(0, config.N - 1), random.randint(0, config.N - 1), random.randint(0, config.K - 1)]
                else:   
                    start_inference = time.time()
                    for robot_key in ['robot_red', 'robot_blue']:
                        core_obs = observations[robot_key]['core_obs'].view(1, -1)
                        ball_obs = observations[robot_key]['ball_obs'].view(1, -1, config.ball_obs_dim)
                        legal_mask = legal_actions[robot_key].view(1, -1)
                        with torch.no_grad():
                            modelout = model(core_obs, ball_obs, legal_mask, inference=True)
                        actions[robot_key] = modelout['action'][0]
                    end_inference = time.time()
                    inference_time += (end_inference - start_inference)
                out = env.step(actions)
                    
            end_time = time.time()
            print(f"Episode(s) finished in {end_time - start_time:.2f} seconds.")
            print(f"Red score: {env.field.red_score}, Blue score: {env.field.blue_score}")
            print(f"Cumulative reward: {cum_reward}")
            print(f"Total action taken by both agents: {env.field.actions_counter  * 2}")
            print(f"Throughput (obs/s): {env.field.actions_counter*2 / (end_time - start_time):.2f}")
            print(f"Total inference time: {inference_time:.2f} seconds.")
            print(f"Average inference time per step: {inference_time / env.field.actions_counter:.4f} seconds.")
            start_time = end_time
    finally:
        env.close()


if __name__ == "__main__":
    torch.set_num_threads(1)
    run_demo(episodes=5)
