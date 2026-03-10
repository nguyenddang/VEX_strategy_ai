from dataclasses import dataclass, field
import math 
import yaml
@dataclass
class VexConfig:
    """Config for this entire repo.
    """
    # ENV Config
    engine_hz: float = 60.0
    inference_hz: float = 20.0
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
    ball_obs_dim: int = 28
    core_obs_dim: int = 85
    n_primary_actions: int = 6
    n_balls: int = 88 # FIXED. never changes
    
    # AGENT Config
    ndim: int = 128 
    
    # TRAINER
    n_workers: int = 2
    buffer_capacity: int = 8192 # can take up 4096 chunks. 
    chunk_size: int = 32 # timesteps per chunk. 
    train_batch_size: int = 8192 # timesteps per training batch. 
    inference_batch_size: int = 512 # number of timestep to inference. 
    inference_timeout: float = 0.001 # max wait time for inference batch. 
    max_league_snapshots: int = 500 
    latest_ratio: float = 0.8 # ratio of workers to use latest snapshot as opponent. 
    inference_grace_period: int = 4 # number of batches to wait before deleting inactive snapshot in inference server.


    steps_per_iteration: int = 32
    inference_server_device: str = 'cuda:1'
    train_device: str = 'cuda:0'
    lr: float = 1e-4
    update_league: int = 10

    # GAE
    gamma: float = 0.99
    lam: float = 0.95

    # LOSS
    value_epsilon: float = 0.2
    policy_epsilon: float = 0.2
    value_coef: float = 1.0
    entropy_coef: float = 0.01
    
    def __post_init__(self):
        assert self.engine_hz >= self.inference_hz, "Engine update frequency should be higher than or equal to inference frequency"
        assert self.engine_hz % self.inference_hz == 0, "Engine update frequency should be divisible by inference frequency"
        assert self.render_hz > 0, "Render frequency must be positive"
        assert self.N % 2 == 1, "MOVE bins (N) should be odd to have a center bin"
        assert 0.5<= self.latest_ratio <= 0.9, "latest_ratio should be between 0.5 and 0.9 to ensure enough diversity in opponents." 
        self.max_actions = int(self.max_duration_s * self.inference_hz)
        self.n_engine_updates = int(self.engine_hz // self.inference_hz)
        self.n_render_updates = max(1, int(self.engine_hz // self.render_hz))
        engine_config_path = 'env/engine_core/config.yml'
        with open(engine_config_path, 'r') as f:
            engine_config = yaml.safe_load(f)
        self.engine_config = engine_config

