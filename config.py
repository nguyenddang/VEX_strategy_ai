from dataclasses import dataclass, field
from pathlib import Path
import math 
import yaml
import os
@dataclass
class VexConfig:
    """Config for this entire repo.
    """
    # ENV Config
    engine_hz: float = 50.0
    inference_hz: float = 5.0
    render_hz: float = 60.0
    realtime_render: bool = True
    max_duration_s: float = 120.0
    max_offset: float = 30.0 # in cm. max relative MOVE extent in x/y from current robot position.
    N: int = 31 # number of bins per axis for MOVE grid around robot.
    K: int = 16 # number of bins for relative heading change in MOVE.
    render_mode: str | None = None
    window_width: int = 1200
    window_height: int = 1200
    robot_capacity: int = 10 # can hold up to robot_capacity balls.
    ball_pickup_hitbox: dict[str, float] = field(
        default_factory=lambda: {
            'dist_threshold': 40, # cm
            'angle_threshold': math.radians(90), 
        }
    ) # robot can pick up ball if satisfy distance and angle thesholds.
    goal_action_hitbox: dict[str, float] = field(
        default_factory=lambda: {
            'dist_threshold': 25,
            'angle_threshold': math.radians(90),
        }
    ) # robot can score if satisfy distance and angle thesholds to the scoring position of the goal.
    loader_pickup_hitbox: dict[str, float] = field(
        default_factory=lambda: {
            'dist_threshold': 25, # cm
            'angle_threshold': math.radians(90), 
        }
    ) # robot can pick up loader if satisfy distance and angle thesholds.
    n_primary_actions: int = 6
    
    # AGENT Config
    ndim: int = 128 
    block_size: int = 32
    
    # TRAINER
    n_workers: int = 56
    buffer_capacity: int = 128 
    train_episodes: int = 32 # timesteps per training batch. 
    mini_train_episodes: int = 16
    max_league_snapshots: int = 1000 
    latest_ratio: float = 0.8 # ratio of workers to use latest snapshot as opponent. 

    steps_per_iteration: int = 8
    train_device: str = 'cuda:0'
    lr: float = 1e-5
    update_league: int = 10

    # GAE
    gamma: float = 0.99
    lam: float = 0.95

    # LOSS
    value_epsilon: float = 0.2
    policy_epsilon: float = 0.2
    value_coef: float = 1.0
    entropy_coef: float = 0.001

    # TRANSFORMER
    compile=True
    n_embd: int = 128
    n_layer: int = 4
    action_size: int = 4
    n_head: int = 4

    core_obs_dim: int = 85
    n_balls: int = 88
    ball_obs_dim: int = 29
    total_timesteps: int = 100

    # LOGGING
    log_wandb: bool = False
    wandb_project: str = "vex-geniusformer-forever"
    wandb_run_name: str = "strategy_without_TS"

    # SAVING CKPTS
    save_ckpt_path: str = "checkpoints"
    n_save_learner_ckpts: int = 500
    n_save_all_ckpts: int = 50

    # LOADING CKPT
    load_ckpt_path: str | None = None
    resume_training: bool = False
    
    def __post_init__(self):
        assert self.engine_hz >= self.inference_hz, "Engine update frequency should be higher than or equal to inference frequency"
        assert self.engine_hz % self.inference_hz == 0, "Engine update frequency should be divisible by inference frequency"
        assert self.render_hz > 0, "Render frequency must be positive"
        assert self.N % 2 == 1, "MOVE bins (N) should be odd to have a center bin"
        assert 0.5<= self.latest_ratio <= 0.9, "latest_ratio should be between 0.5 and 0.9 to ensure enough diversity in opponents." 
        self.max_actions = int(self.max_duration_s * self.inference_hz)
        self.total_timesteps = self.max_actions
        self.n_engine_updates = int(self.engine_hz // self.inference_hz)
        self.n_render_updates = max(1, int(self.engine_hz // self.render_hz))
        engine_config_path = 'env/engine_core/config.yml'
        with open(engine_config_path, 'r') as f:
            engine_config = yaml.safe_load(f)
        self.engine_config = engine_config

        # creating ckpt directory if not exist
        os.makedirs(self.save_ckpt_path, exist_ok=True)
