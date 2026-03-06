from pathlib import Path
import random
import sys
import time

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from env.env import VexEnv
from env.config import EnvConfig
import yaml 

engine_config_path = 'env/engine_core/config.yml'
with open(engine_config_path, 'r') as f:
    engine_config = yaml.safe_load(f)

def run_demo(episodes: int = 1) -> None:
    env = VexEnv(
        EnvConfig(
            render_mode=None,
            engine_hz=60.0,
            inference_hz=5.0,
            max_duration_s=120.0,
            render_hz=30.0,
            realtime_render=False
        ),
        engine_config=engine_config,
    )
    start_time = time.time()
    try:
        for _ in range(episodes):
            out = env.reset()
            cum_reward = {'robot_red': 0.0, 'robot_blue': 0.0}
            while not out["done"]:
                action = {}
                legal_actions = out["legal_actions"]
                action = {}
                for player in ["robot_red", "robot_blue"]:
                    legal_move = [i for i, valid in enumerate(legal_actions[player]) if valid]
                    action[player] = [random.choice(legal_move)]
                    move_x = random.randint(0, env.env_config.N - 1)
                    move_y = random.randint(0, env.env_config.N - 1)
                    move_theta = random.randint(0, env.env_config.K - 1)
                    action[player].extend([move_x, move_y, move_theta])
                out = env.step(action)
                cum_reward['robot_red'] += out['reward']['robot_red']
                cum_reward['robot_blue'] += out['reward']['robot_blue']
    finally:
        env.close()

    end_time = time.time()
    print(f"Episode(s) finished in {end_time - start_time:.2f} seconds.")
    print(f"Average time per episode: {(end_time - start_time) / episodes:.2f} seconds.")
    print(f"Red score: {env.field.red_score}, Blue score: {env.field.blue_score}")
    print(f"Cumulative reward: {cum_reward}")

if __name__ == "__main__":
    run_demo(episodes=1)
