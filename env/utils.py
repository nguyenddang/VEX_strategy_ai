from env.config import EnvConfig
from env.engine_core.field_component import Ball, Goal, Loader, Loader_Manager, Leg, Wall
from env.engine_core.robot import Robot
from env.type import Field

from typing import Any, Dict, Tuple, Callable
import pymunk
import torch 
import time

def create_space() -> pymunk.Space:
    space = pymunk.Space()
    space.gravity = (0, 0)
    space.collision_slop = 0.0
    space.idle_speed_threshold = 1.0
    space.sleep_time_threshold = 0.6
    space.iterations = 10
    return space

def step_space(space: pymunk.Space, dt: float) -> None:
    substeps = 4
    sub_dt = dt / substeps
    for _ in range(substeps):
        space.step(sub_dt)
        
def build_world(engine_config: Dict[str, Any]) -> Tuple[pymunk.Space, Field]:
    """Build ... the WORLD. WOAH!
    """
    field_config = engine_config["field"]
    loader_config = engine_config["loader"]
    goal_config = engine_config["goal"]
    robot_config = engine_config["robot"]
    leg_config = engine_config["leg"]
    wall_config = engine_config["wall"]
    ball_config = engine_config["ball"]
    blocking_config = engine_config["blocking"]
    space = create_space()
    
    # Loaders and Loader Managers
    loader_manager12 = Loader_Manager(space, colour="red", ball_config=ball_config, debug=False)
    loader_manager34 = Loader_Manager(space, colour="blue", ball_config=ball_config, debug=False)
    loaders = []
    for key in ["loader_1", "loader_2", "loader_3", "loader_4"]:
        lm = loader_manager12 if key in ["loader_1", "loader_2"] else loader_manager34
        loader = Loader(space, loader_config=loader_config[key], robot_config=robot_config, ball_config=ball_config, manager=lm, key=key)
        loaders.append(loader)
        
    # Goals 
    goals = []
    for key in ["center_upper", "center_lower", "long_1", "long_2"]:
        goal = Goal(space, goal_config[key], blocking_config, ball_config, robot_config, key=key, debug=False)
        goals.append(goal)
        
    # Robots
    robot_red = Robot(space, robot_config=robot_config, key="robot_red", field_config=field_config, ball_config=ball_config)
    robot_blue = Robot(space, robot_config=robot_config, key="robot_blue", field_config=field_config, ball_config=ball_config) 
        
    # Balls on field 
    balls = []
    for colour in ["red", "blue"]:
        for position in engine_config["initial_ground_ball_positions"][colour]:
            ball = Ball(space, colour=colour, ball_config=ball_config, state="ground", add_sim=True, position=position)
            balls.append(ball)
    balls.extend(loader_manager12.inventory)
    balls.extend(loader_manager34.inventory)
    balls.extend(robot_red.inventory)
    balls.extend(robot_blue.inventory)
    for loader in loaders:
        balls.extend(loader.scored_balls)
    
    # Walls and Legs
    wall = Wall(space, wall_config, field_config)
    legs = []
    for key in ["leg1", "leg2", "leg3", "leg4"]:
        leg = Leg(space, leg_config[key], key=key)
        legs.append(leg)
        
    field = Field(
        wall=wall,
        legs=legs,
        goals=goals,
        loaders=loaders,
        robot_red=robot_red,
        robot_blue=robot_blue,
        balls=balls,
        actions_counter=0,
    )
    return space, field


def update_world(
    field: Field,
    n_engine_updates: int,
    engine_update_dt: float,
    n_render_updates: int,
    step_space_fn: Callable[[Any, float], None],
    render_fn: Callable[[], None] | None = None,
    realtime: bool = False,
) -> None:
    space = field.robot_red.space
    next_step_wall_time = time.perf_counter()
    for istep in range(n_engine_updates):
        if realtime:
            now = time.perf_counter()
            sleep_for = next_step_wall_time - now
            if sleep_for > 0.0:
                time.sleep(sleep_for)
        field.robot_red.update(engine_update_dt)
        field.robot_blue.update(engine_update_dt)
        step_space_fn(space, engine_update_dt)
        if render_fn is not None and istep % n_render_updates == 0:
            render_fn()
        if realtime:
            next_step_wall_time += engine_update_dt
            
def get_match_score(field: Field) -> tuple[int, int]:
    red_total = 0
    blue_total = 0
    for goal in field.goals:
        red_goal_score, blue_goal_score = goal.get_game_score()
        red_total += red_goal_score
        blue_total += blue_goal_score
    return red_total, blue_total
