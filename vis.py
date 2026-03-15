
from networkx import config
import torch
from config import VexConfig
from env.env import VexEnv
from model.model import GeniusFormer
import time
import math 
def main():
	# Load config and environment
	checkpoint = torch.load("checkpoints/learner_1500.pt", map_location="cpu")
	config = VexConfig()
	config_dict = checkpoint['config']
	# load config values from checkpoint
	for key, value in config_dict.items():
		setattr(config, key, value)
	config.render_mode = 'human'
	config.loader_pickup_hitbox = {
		'dist_threshold': 35, # cm
		'angle_threshold': math.radians(45),
	}
	config.goal_action_hitbox = {
		'dist_threshold': 25,
		'angle_threshold': math.radians(45),
	}
	config.ball_pickup_hitbox = {
		'dist_threshold': 25, # cm
		'angle_threshold': math.radians(45),
	}
	env = VexEnv(config)
	model = GeniusFormer(config)
	model.load_state_dict(checkpoint['model'])
	model.eval()

	obs = env.reset()
	done = obs.get('done', False)
	timestep = 0
	while not done:
		# Use model for inference
		obs_dict = obs['observations']
		legal_actions = obs['legal_actions']

		# Prepare input tensors for both robots
		actions = {}
		for robot_key in ['robot_red', 'robot_blue']:
			core_obs = obs_dict[robot_key]['core_obs'].unsqueeze(0).unsqueeze(0)  # (B=1, T=1, core_obs_dim)
			ball_obs = obs_dict[robot_key]['ball_obs'].unsqueeze(0).unsqueeze(0)  # (B=1, T=1, n_balls, ball_obs_dim)
			legal_mask = legal_actions[robot_key].unsqueeze(0)  # (B=1, T=1, n_primary_actions)
			with torch.no_grad():
				out = model(core_obs, ball_obs, legal_mask, do_inference=True)
				act = out['actions'][0, :].cpu().numpy().tolist()  # [discrete, x, y, theta]
			actions[robot_key] = act

		obs = env.step(actions)
		done = obs.get('done', False)
		timestep += 1
		time.sleep(1.0 / config.render_hz)

	env.close()

if __name__ == "__main__":
	main()
