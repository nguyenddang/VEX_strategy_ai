
from dataclasses import dataclass, field
import math 
@dataclass
class EnvConfig:
    engine_hz: float = 60.0
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