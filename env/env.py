from dataclasses import dataclass, field
from typing import Any, Dict
import math
import random
import time

try:
    from engine import *
except ImportError:
    from env.engine import *


def _create_renderer(**kwargs):
    try:
        from renderer import EnvRenderer
    except ImportError:
        from env.renderer import EnvRenderer
    return EnvRenderer(**kwargs)


@dataclass
class Config:
    engine_hz: float = 60.0
    env_hz: float = 5.0
    render_hz: float = 30.0
    max_duration_s: float = 120.0
    max_offset: float = 30.0 # in cm. max distance from current position for the MOVE action. This is used to scale the continuous action inputs.
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


class VexEnv:

    def __init__(self, config: Config | None = None):
        self.config = config or Config()
        self.engine_update_dt = 1.0 / self.config.engine_hz
        self.model_query_dt = 1.0 / self.config.env_hz
        self.render_update_dt = 1.0 / self.config.render_hz if self.config.render_hz > 0 else None
        self._build_world()
        self.sim_time_s = 0.0
        self.decision_elapsed_s = 0.0
        self.render_elapsed_s = 0.0
        self.done = False
        self.renderer = None
        self.total_actions = 0

        if self.config.render_mode == "human":
            self.renderer = _create_renderer(
                window_width=self.config.window_width,
                window_height=self.config.window_height,
                pickup_dist_threshold=self.config.ball_pickup_hitbox["dist_threshold"],
                pickup_angle_threshold_deg=math.degrees(self.config.ball_pickup_hitbox["angle_threshold"]),
                goal_dist_threshold=self.config.goal_action_hitbox["dist_threshold"],
                goal_angle_threshold_deg=math.degrees(self.config.goal_action_hitbox["angle_threshold"]),
                loader_dist_threshold=self.config.loader_pickup_hitbox["dist_threshold"],
            )

    def _build_world(self) -> None:
        self.space = create_space()
        loader_manager12 = Loader_Manager(self.space, colour="red")
        loader_manager34 = Loader_Manager(self.space, colour="blue")
        loader_1 = Loader(self.space, position=LOADER_1_POSITION, goal_key="loader_1", initial_ball_codes=INITIAL_LOADERS_BALLS[0], manager=loader_manager12)
        loader_2 = Loader(self.space, position=LOADER_2_POSITION, goal_key="loader_2", initial_ball_codes=INITIAL_LOADERS_BALLS[1], manager=loader_manager12)
        loader_3 = Loader(self.space, position=LOADER_3_POSITION, goal_key="loader_3", initial_ball_codes=INITIAL_LOADERS_BALLS[2], manager=loader_manager34)
        loader_4 = Loader(self.space, position=LOADER_4_POSITION, goal_key="loader_4", initial_ball_codes=INITIAL_LOADERS_BALLS[3], manager=loader_manager34)
        loader_list = [loader_1, loader_2, loader_3, loader_4]

        loader_balls = []
        for loader in loader_list:
            loader_balls.extend([ball for ball in loader.scored_balls if ball is not None])

        ground_balls = [
            Ball(self.space, position=position, colour="red")
            for position in INITIAL_BALL_POSITIONS_CM["red"]
        ] + [
            Ball(self.space, position=position, colour="blue")
            for position in INITIAL_BALL_POSITIONS_CM["blue"]
        ]

        red_robot = Robot(
            self.space,
            position=INITIAL_RED_ROBOT_POSITION_CM,
            angle=math.radians(INITAL_RED_ROBOT_ANGLE_DEG),
            colour="red",
            capacity=self.config.robot_capacity,
        )
        blue_robot = Robot(
            self.space,
            position=INITIAL_BLUE_ROBOT_POSITION_CM,
            angle=math.radians(INITAL_BLUE_ROBOT_ANGLE_DEG),
            colour="blue",
            capacity=self.config.robot_capacity,
        )
        tracked_balls = ground_balls + loader_balls + red_robot.inventory + blue_robot.inventory + loader_manager12.inventory + loader_manager34.inventory

        self.field_dict = {
            "wall": Wall(self.space),
            "CGU": Goal(self.space, position=CENTER_GOAL_POSITION, length=CENTER_GOAL_UPPER_LENGTH, width=CENTER_GOAL_UPPER_WIDTH, angle_deg=-45, goal_key='center_upper', capacity=7, shape_category=CATEGORY_GOAL_UPPER, mask=CATEGORY_ROBOT),
            "CGL": Goal(self.space, position=CENTER_GOAL_POSITION, length=CENTER_GOAL_LOWER_LENGTH, width=CENTER_GOAL_LOWER_WIDTH, angle_deg=45, goal_key='center_lower', capacity=7),
            "LG1": Goal(self.space, position=LONG_GOAL_1_POSITION, length=LONG_GOAL_LENGTH, width=LONG_GOAL_WIDTH, angle_deg=0, goal_key='long_1', capacity=15, shape_category=CATEGORY_GOAL_UPPER, mask=CATEGORY_ROBOT),
            "LG2": Goal(self.space, position=LONG_GOAL_2_POSITION, length=LONG_GOAL_LENGTH, width=LONG_GOAL_WIDTH, angle_deg=0, goal_key='long_2', capacity=15, shape_category=CATEGORY_GOAL_UPPER, mask=CATEGORY_ROBOT),
            "legs": [
                Leg(self.space, position=(LONG_GOAL_LEG_X1, LONG_GOAL_1_POSITION[1])),
                Leg(self.space, position=(LONG_GOAL_LEG_X2, LONG_GOAL_1_POSITION[1])),
                Leg(self.space, position=(LONG_GOAL_LEG_X1, LONG_GOAL_2_POSITION[1])),
                Leg(self.space, position=(LONG_GOAL_LEG_X2, LONG_GOAL_2_POSITION[1])),
            ],
            "LD1": loader_1,
            "LD2": loader_2,
            "LD3": loader_3,
            "LD4": loader_4,
            "loaders": loader_list,
            "red_robot": red_robot,
            "blue_robot": blue_robot,
            "balls": tracked_balls,
        }

    def reset(self) -> Dict[str, Any]:
        self._build_world()
        self.sim_time_s = 0.0
        self.decision_elapsed_s = 0.0
        self.render_elapsed_s = 0.0
        self.done = False
        if self.renderer is not None:
            self.render()
        self._update_ball_relative_states()
        # TODO: return initial observation
        return {}

    def step(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """Step env

        Args:
            action : Action dict for both red and blue. Each contain 6 discrete and 4 continuous. 
            
            Discrete: 
            - 0: NO-OP
            - 1: MOVE
            - 2: PICKUP LOADERS
            - 3: PICKUP GROUND
            - 4: SCORE
            - 5: BLOCK
            
            Continuous:
            - delta_x: [-1, 1] offset from current position, scaled by self.config.max_offset
            - delta_y: [-1, 1] offset from current position, scaled by self.config.max_offset
            - sin(delta_theta): relative turn command
            - cos(delta_theta): relative turn command
            
            Note: Continuous inputs are only used for MOVE action. 
        """
        
        self.total_actions += 1
        for player in ["red", "blue"]:
            dis_act = action[player]["discrete"]
            cont_act = action[player]["continuous"]
            robot = self.field_dict[f"{player}_robot"]

            if dis_act == 0: # NO-OP
                pass
            elif dis_act == 1: # MOVE
                delta_x, delta_y, sin_theta, cos_theta = cont_act
                target_x = robot.body.position.x + delta_x * self.config.max_offset
                target_y = robot.body.position.y + delta_y * self.config.max_offset
                target_theta = robot.body.angle + math.atan2(sin_theta, cos_theta)
                robot.set_motion_target((target_x, target_y), target_theta)
            elif dis_act == 2: # PICKUP LOADERS
                robot.pickup_loader()
            elif dis_act == 3: # PICKUP GROUND
                robot.pickup_ground()
            elif dis_act == 4: # SCORE
                robot.score_goal()
            elif dis_act == 5: # BLOCK
                robot.block_goal()
        
        self.update_world()
        for player in ["red", "blue"]:
            robot = self.field_dict[f"{player}_robot"]
            robot.clear_action_attempt()
        self._sync_inventory_ball_positions()
        self._update_ball_relative_states()
        legal_actions = self.get_legal_actions()
        return {'legal_actions': legal_actions} # TODO: also return observation and reward

    def _sync_inventory_ball_positions(self):
        for player in ["red", "blue"]:
            robot = self.field_dict[f"{player}_robot"]
            for ball in robot.inventory:
                ball.body.position = robot.body.position

    def _update_ball_relative_states(self):
        """Update all balls' relative states to both robot and check if can be picked up.
        If multiple ball can be picked up, pick the closest one.
        """
        min_r, min_b = float("inf"), float("inf")
        dist_threshold, angle_threshold = self.config.ball_pickup_hitbox["dist_threshold"], self.config.ball_pickup_hitbox["angle_threshold"]
        red_robot, blue_robot = self.field_dict["red_robot"], self.field_dict["blue_robot"]
        red_robot._pickup_ball, blue_robot._pickup_ball = None, None
        for ball in self.field_dict["balls"]:
            relative_r, relative_blue = ball.update_relative_to_robot(red_robot, blue_robot)
            # check if ball can be picked up. If not on ground, ignore. 
            if ball.state != "ground":
                continue
            ball_colour = ball.colour
            relative = relative_r if ball_colour == "red" else relative_blue
            robot = red_robot if ball_colour == "red" else blue_robot
            can_pickup = relative["distance"] <= dist_threshold and abs(relative["delta_theta"]) <= angle_threshold and len(robot.inventory) <= robot.capacity
            if can_pickup:
                if relative["distance"] < min_r and ball_colour == "red":
                    red_robot._pickup_ball = ball
                    min_r = relative["distance"]
                elif relative["distance"] < min_b and ball_colour == "blue":
                    blue_robot._pickup_ball = ball
                    min_b = relative["distance"]

    def update_world(self):
        while self.decision_elapsed_s < self.model_query_dt:
            tick_start = time.perf_counter()
            self.field_dict["red_robot"].update(self.engine_update_dt)
            self.field_dict["blue_robot"].update(self.engine_update_dt)
            step_space(self.space, self.engine_update_dt)
            self.render_elapsed_s += self.engine_update_dt

            if self.renderer is not None:
                if self.render_update_dt is not None and self.render_elapsed_s >= self.render_update_dt:
                    self.render()
                    self.render_elapsed_s = 0.0

                elapsed_tick = time.perf_counter() - tick_start
                sleep_time = self.engine_update_dt - elapsed_tick
                if sleep_time > 0:
                    time.sleep(sleep_time)

            self.decision_elapsed_s += self.engine_update_dt
            self.sim_time_s += self.engine_update_dt
            if self.sim_time_s >= self.config.max_duration_s:
                self.done = True
                break
        self.decision_elapsed_s = 0.0

    def _find_nearest_building_target(self, robot, opponent_robot, goals):
        """ Find goal to score or block based on proximity and angle to the scoring positions.
        Assumes scoring/blocking regions do not overlap. Therefore at most one scoring and one blocking target exist, 
        and is the same goal.
        """
        goal_dist_threshold = self.config.goal_action_hitbox["dist_threshold"]
        goal_angle_threshold = self.config.goal_action_hitbox["angle_threshold"]
        blocker_pos = (opponent_robot.body.position.x, opponent_robot.body.position.y)

        # assumes goal action hitboxes do not overlap: at most one scoring region matches.
        for goal in goals:
            for side_index, scoring_position in zip(goal.score_side, goal.scoring_position):
                dx = scoring_position[0] - robot.body.position.x
                dy = scoring_position[1] - robot.body.position.y
                target_distance = math.hypot(dx, dy)
                target_heading = math.atan2(
                    goal.body.position.y - scoring_position[1],
                    goal.body.position.x - scoring_position[0],
                )
                heading_error = robot._normalize_angle(target_heading - robot.body.angle)
                within_goal_hitbox = (
                    target_distance <= goal_dist_threshold and abs(heading_error) <= goal_angle_threshold
                )
                if not within_goal_hitbox:
                    continue

                block_target = (scoring_position, target_heading, goal, side_index, opponent_robot)
                if goal.can_accept(entry_side=side_index, blocker_pos=blocker_pos):
                    score_target = (scoring_position, target_heading, goal, side_index, opponent_robot)
                    return score_target, block_target
                return None, block_target

        return None, None

    def _find_nearest_loader_pickup_target(self, robot, loaders):
        loader_dist_threshold = self.config.loader_pickup_hitbox["dist_threshold"]
        loader_angle_threshold = self.config.loader_pickup_hitbox["angle_threshold"]
        for loader in loaders:
            if loader.scored_balls[0] is None:
                continue
            loading_position = loader.loading_position
            dx = loading_position[0] - robot.body.position.x
            dy = loading_position[1] - robot.body.position.y
            target_distance = math.hypot(dx, dy)
            target_heading = math.atan2(
                loader.body.position.y - loading_position[1],
                loader.body.position.x - loading_position[0],
            )
            heading_error = robot._normalize_angle(target_heading - robot.body.angle)
            within_loader_hitbox = (
                target_distance <= loader_dist_threshold and abs(heading_error) <= loader_angle_threshold
            )
            if not within_loader_hitbox:
                continue
            return (loading_position, target_heading, loader)
        return None
        
    def get_legal_actions(self) -> Dict[str, Any]:
        """Return legal discrete actions for both players.

        Discrete action indices: [NO-OP, MOVE, PICKUP LOADERS, PICKUP GROUND, SCORE, BLOCK]
        Continuous actions are always legal.
        """
        legal_actions: Dict[str, Any] = {}
        goals = [self.field_dict["CGL"], self.field_dict["CGU"], self.field_dict["LG1"], self.field_dict["LG2"]]

        for player in ["red", "blue"]:
            robot_key = f"{player}_robot"
            robot = self.field_dict[robot_key]
            opponent_robot = self.field_dict["blue_robot"] if player == "red" else self.field_dict["red_robot"]
            score_target, block_target = self._find_nearest_building_target(
                robot=robot,
                opponent_robot=opponent_robot,
                goals=goals,
            )
            loader_target = self._find_nearest_loader_pickup_target(robot, self.field_dict["loaders"])
            if len(robot.inventory) == 0:
                score_target = None
            can_score = score_target is not None
            can_block = block_target is not None
            can_pickup_loader = loader_target is not None
            can_pickup_ground = robot._pickup_ball is not None and len(robot.inventory) < robot.capacity
            discrete_mask = [True, True, can_pickup_loader, can_pickup_ground, can_score, can_block]
            legal_actions[player] = {
                "discrete_mask": discrete_mask,
            }
            robot._building_score_target = score_target
            robot._building_block_target = block_target
            robot._building_loader_target = loader_target
        return legal_actions
        
    def render(self):
        if self.renderer is None:
            return None
        return self.renderer.render(self.field_dict)

    def close(self):
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None

import time 
if __name__ == "__main__":
    env = VexEnv(Config(render_mode=None, engine_hz=50.0, env_hz=5.0, max_duration_s=120.0))
    start_time = time.time()
    for i in range(1):
        env.reset()
        while not env.done:
            action = {}
            legal_actions = env.get_legal_actions()
            for player in ["red", "blue"]:
                theta = random.uniform(-math.pi, math.pi)
                possible_actions = [i for i in range(1, 6) if legal_actions[player]["discrete_mask"][i]]
                da = random.choice(possible_actions) 
                action[player] = {
                    "discrete": da,
                    "continuous": [
                        random.uniform(-1.0, 1.0),
                        random.uniform(-1.0, 1.0),
                        math.sin(theta),
                        math.cos(theta),
                    ],
                }
            env.step(action)
    env.close()
    end_time = time.time()
    print(f"Episode finished in {end_time - start_time:.2f} seconds.")
    print(f"Total actions taken: {env.total_actions}")

