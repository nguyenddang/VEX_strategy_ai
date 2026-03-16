
from networkx import config
import torch
from config import VexConfig
from env.env import VexEnv
from model.mlp import MLP
import time
import math 
def main():
	# Load config and environment
	checkpoint_learner = torch.load("checkpoints/learner_2750.pt", map_location="cpu")
	checkpoint_opp = torch.load("checkpoints/learner_2750.pt", map_location="cpu")
	config = VexConfig()
	config_dict = checkpoint_learner['config']
	# load config values from checkpoint
	for key, value in config_dict.items():
		setattr(config, key, value)
	config.render_mode = 'human'
	config.realtime_render = False
	env = VexEnv(config)
	learner = MLP(config)
	learner.load_state_dict(checkpoint_learner['model'])
	learner.eval()
	opp = MLP(config)
	opp.load_state_dict(checkpoint_opp['model'])
	opp.eval()

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
			core_obs = obs_dict[robot_key]['core_obs'].view(1, -1)
			ball_obs = obs_dict[robot_key]['ball_obs'].view(1, config.n_balls, -1)  # (B=1, T=1, n_balls, ball_obs_dim)
			legal_mask = legal_actions[robot_key].view(1, -1)  # (B=1, T=1, n_primary_actions)
			with torch.no_grad():
				model = learner if robot_key == 'robot_red' else opp
				out = model(core_obs, ball_obs, legal_mask, inference=True)
				act = out['actions'][0, :].tolist()  # [discrete, x, y, theta]
			actions[robot_key] = act

		obs = env.step(actions)
		done = obs.get('done', False)
		timestep += 1
		time.sleep(1.0 / config.render_hz)

	env.close()

if __name__ == "__main__":
	main()
