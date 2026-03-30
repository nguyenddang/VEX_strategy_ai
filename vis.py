
from networkx import config
import torch
from config import VexConfig
from env.env import VexEnv
from model.mlp import MLP
import time
import math 
def main():
	# Load config and environment
	checkpoint_learner = torch.load("checkpoints_7/learner_18750.pt", map_location="cpu")
	checkpoint_opp = torch.load("checkpoints_7/learner_16500.pt", map_location="cpu")
	config = VexConfig()
	config_dict = checkpoint_learner['config']
	# load config values from checkpoint
	for key, value in config_dict.items():
		setattr(config, key, value)
	config.window_height = 1000
	config.window_width = 1000
	config.render_mode = None
	config.realtime_render = False
	env = VexEnv(config)
	learner = MLP(config)
	m, n, = 0.0, 0.0
	t = 0.0
	unwanted_prefix = '_orig_mod.'
	for k,v in list(checkpoint_learner['model'].items()):
		if k.startswith(unwanted_prefix):
			checkpoint_learner['model'][k[len(unwanted_prefix):]] = checkpoint_learner['model'].pop(k)
	for k,v in list(checkpoint_opp['model'].items()):
		if k.startswith(unwanted_prefix):
			checkpoint_opp['model'][k[len(unwanted_prefix):]] = checkpoint_opp['model'].pop(k)
	learner.load_state_dict(checkpoint_learner['model'])
	learner.eval()
	opp = MLP(config)
	opp.load_state_dict(checkpoint_opp['model'])
	opp.eval()

	env_out = env.reset()
	done = env_out.get('done', False)
	while not done:
		# Use model for inference
		obs_dict = env_out['observations']
		legal_actions = env_out['legal_actions']

		# Prepare input tensors for both robots
		actions = {}
		learner_out = learner(obs_dict['robot_red']['core_obs'].view(1, -1), 
							obs_dict['robot_red']['ball_obs'].view(1, config.n_balls, -1), 
							legal_actions['robot_red'].view(1, -1), inference=True, argmax=False)
		opp_out = opp(obs_dict['robot_blue']['core_obs'].view(1, -1), 
						obs_dict['robot_blue']['ball_obs'].view(1, config.n_balls, -1), 
						legal_actions['robot_blue'].view(1, -1), inference=True, argmax=False)
		actions = {
			'robot_red': learner_out['actions'][0, :].clone().tolist(),  # [discrete, x, y, theta]
			'robot_blue': opp_out['actions'][0, :].clone().tolist()  # [discrete, x, y, theta]
		}
		if actions['robot_red'][0] == 0:
			n += 1
		if actions['robot_red'][0] != 0: 
			m += 1
			t += n
			n = 0.0
		env_out = env.step(actions)
		done = env_out.get('done', False)
	print(f"{'move': m, 'total': t, 'avg': t/m if m > 0 else 0.0}")

	env.close()

if __name__ == "__main__":
	main()
