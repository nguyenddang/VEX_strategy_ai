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
    inference_hz: float = 5.0
    render_hz: float = 60.0
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


class VexEnv:

    def __init__(self, config: Config | None = None):
        self.config = config or Config()
        assert config.engine_hz >= config.inference_hz, "Engine update frequency should be higher than or equal to inference frequency"
        assert config.engine_hz % config.inference_hz == 0, "Engine update frequency should be divisible by inference frequency"
        self.n_engine_updates = int(config.engine_hz // config.inference_hz)
        self.n_render_updates = int(config.engine_hz // config.render_hz)
        self.max_actions = config.max_duration_s * config.inference_hz
        self.engine_update_dt = 1.0 / config.engine_hz
        self.action_count = 0
        self._build_world()
        self.renderer = None
        self.time_temp = {}

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
            "elapsed_time_s": self.action_count / self.config.inference_hz,
            "max_duration_s": self.config.max_duration_s,
        }

    def _update_time_metadata(self):
        self.field_dict["elapsed_time_s"] = self.action_count / self.config.inference_hz
        self.field_dict["max_duration_s"] = self.config.max_duration_s

    def reset(self) -> Dict[str, Any]:
        self.action_count = 0
        self._build_world()
        self._update_time_metadata()
        done = False
        if self.renderer is not None:
            self.render()
        self._update_ball_relative_states()
        # TODO: return initial observation
        return {'done': done}

    def step(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """Step env

        Args:
            action : Action dict for both red and blue.
            Required keys per player:
            - discrete: action type index [0..5]
            - move_x: MOVE x-bin index [0..N-1]
            - move_y: MOVE y-bin index [0..N-1]
            - move_theta: MOVE heading-bin index [0..K-1]
            
            Discrete: 
            - 0: NO-OP
            - 1: MOVE
            - 2: PICKUP LOADERS
            - 3: PICKUP GROUND
            - 4: SCORE
            - 5: BLOCK

            MOVE bins are only used when discrete == 1.
        """
        step_start_time = time.time()
        self.action_count += 1
        self._update_time_metadata()
        for player in ["red", "blue"]:
            dis_act = action[player]["discrete"]
            robot = self.field_dict[f"{player}_robot"]

            if dis_act == 0: # NO-OP
                pass
            elif dis_act == 1: # MOVE
                x_idx = action[player]["move_x"]
                y_idx = action[player]["move_y"]
                theta_idx = action[player]["move_theta"]

                delta_x = -self.config.max_offset + (2.0 * self.config.max_offset * x_idx) / (self.config.N - 1)
                delta_y = -self.config.max_offset + (2.0 * self.config.max_offset * y_idx) / (self.config.N - 1)
                delta_theta = -math.pi + (2.0 * math.pi * (theta_idx + 0.5)) / self.config.K

                target_x = robot.body.position.x + delta_x
                target_y = robot.body.position.y + delta_y
                target_theta = robot.body.angle + delta_theta
                robot.set_motion_target((target_x, target_y), target_theta)
            elif dis_act == 2: # PICKUP LOADERS
                robot.pickup_loader()
            elif dis_act == 3: # PICKUP GROUND
                robot.pickup_ground()
            elif dis_act == 4: # SCORE
                robot.score_goal()
            elif dis_act == 5: # BLOCK
                robot.block_goal()
        
        update_engine_start_time = time.time()
        self.update_world()
        update_engine_end_time = time.time()
        for player in ["red", "blue"]:
            robot = self.field_dict[f"{player}_robot"]
            robot.clear_action_attempt()
        self._sync_inventory_ball_positions()
        relative_start_time = time.time()
        self._update_ball_relative_states()
        relative_end_time = time.time()
        get_legal_start_time = time.time()
        legal_actions = self.get_legal_actions()
        get_legal_end_time = time.time()
        self.time_temp["get_legal_actions_time"] = self.time_temp.get("get_legal_actions_time", 0) + (get_legal_end_time - get_legal_start_time)
        done = self.action_count >= self.max_actions
        step_end_time = time.time()
        self.time_temp["step_time"] = self.time_temp.get("step_time", 0) + (step_end_time - step_start_time)
        self.time_temp["engine_update_time"] = self.time_temp.get("engine_update_time", 0) + (update_engine_end_time - update_engine_start_time)
        self.time_temp["relative_update_time"] = self.time_temp.get("relative_update_time", 0) + (relative_end_time - relative_start_time)
        red_score, blue_score = self.get_match_score()
        return {'legal_actions': legal_actions, 'done': done, 'red_score': red_score, 'blue_score': blue_score} # TODO: also return observation and reward

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
            can_pickup = relative["distance"] <= dist_threshold and abs(relative["delta_theta"]) <= angle_threshold and len(robot.inventory) < robot.capacity
            if can_pickup:
                if relative["distance"] < min_r and ball_colour == "red":
                    red_robot._pickup_ball = ball
                    min_r = relative["distance"]
                elif relative["distance"] < min_b and ball_colour == "blue":
                    blue_robot._pickup_ball = ball
                    min_b = relative["distance"]

    def update_world(self):
        for istep in range(self.n_engine_updates):
            self.field_dict["red_robot"].update(self.engine_update_dt)
            self.field_dict["blue_robot"].update(self.engine_update_dt)
            step_space(self.space, self.engine_update_dt)

            if self.renderer is not None and istep % self.n_render_updates == 0:
                self.render()

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
            can_pickup_ground = robot._pickup_ball is not None 
            discrete_mask = [True, True, can_pickup_loader, can_pickup_ground, can_score, can_block]
            legal_actions[player] = {
                "discrete_mask": discrete_mask,
            }
            robot._building_score_target = score_target
            robot._building_block_target = block_target
            robot._building_loader_target = loader_target
        return legal_actions

    def get_match_score(self) -> tuple[int, int]:
        goals = [self.field_dict["CGL"], self.field_dict["CGU"], self.field_dict["LG1"], self.field_dict["LG2"]]
        red_total = 0
        blue_total = 0
        for goal in goals:
            red_goal_score, blue_goal_score = goal.get_game_score()
            red_total += red_goal_score
            blue_total += blue_goal_score
        return red_total, blue_total
        
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
    env = VexEnv(Config(render_mode=None, engine_hz=60.0, inference_hz=5.0, max_duration_s=120.0, render_hz=30.0))
    start_time = time.time()
    for i in range(1):
        out = env.reset()
        while not out['done']:
            action = {}
            legal_actions = env.get_legal_actions()
            for player in ["red", "blue"]:
                possible_actions = [i for i in range(1, 6) if legal_actions[player]["discrete_mask"][i]]
                da = random.choice(possible_actions) 
                action[player] = {
                    "discrete": da,
                    "move_x": random.randrange(env.config.N),
                    "move_y": random.randrange(env.config.N),
                    "move_theta": random.randrange(env.config.K),
                }
            out = env.step(action)
        print(f"Final Score - Red: {out['red_score']}, Blue: {out['blue_score']}")
    env.close()
    end_time = time.time()
    print(f"Episode finished in {end_time - start_time:.2f} seconds.")
    print(f"Total actions taken: {env.action_count}")
    print(f"Total step time: {env.time_temp.get('step_time', 0):.2f} seconds.")
    print(f"Total engine update time: {env.time_temp.get('engine_update_time', 0):.2f} seconds.")
    print(f"Total get_legal_actions time: {env.time_temp.get('get_legal_actions_time', 0):.2f} seconds.")
    print(f"Total relative state update time: {env.time_temp.get('relative_update_time', 0):.2f} seconds.")