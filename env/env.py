from dataclasses import dataclass, field
from typing import Any, Dict
import math
import random
import time

try:
    from renderer import EnvRenderer
except ImportError:
    from env.renderer import EnvRenderer

try:
    from engine import (
        INITIAL_BALL_POSITIONS_CM,
        INITIAL_BLUE_ROBOT_POSITION_CM,
        INITIAL_RED_ROBOT_POSITION_CM,
        INITAL_BLUE_ROBOT_ANGLE_DEG,
        INITAL_RED_ROBOT_ANGLE_DEG,
        BALL_RADIUS,
        ROBOT_SIZE,
        Ball,
        CenterGoalLower,
        CenterGoalUpper,
        LongGoalOne,
        LongGoalOneLegOne,
        LongGoalOneLegTwo,
        LongGoalTwo,
        LongGoalTwoLegOne,
        LongGoalTwoLegTwo,
        Robot,
        Wall,
        create_space,
        step_space,
    )
except ImportError:
    from env.engine import (
    INITIAL_BALL_POSITIONS_CM,
    INITIAL_BLUE_ROBOT_POSITION_CM,
    INITIAL_RED_ROBOT_POSITION_CM,
    INITAL_BLUE_ROBOT_ANGLE_DEG,
    INITAL_RED_ROBOT_ANGLE_DEG,
    BALL_RADIUS,
    ROBOT_SIZE,
    Ball,
    CenterGoalLower,
    CenterGoalUpper,
    LongGoalOne,
    LongGoalOneLegOne,
    LongGoalOneLegTwo,
    LongGoalTwo,
    LongGoalTwoLegOne,
    LongGoalTwoLegTwo,
    Robot,
    Wall,
    create_space,
    step_space,
    )


@dataclass
class Config:
    engine_hz: float = 60.0
    env_hz: float = 5.0
    render_hz: float = 60.0
    max_duration_s: float = 120.0
    max_offset: float = 30.0 # in cm. max distance from current position for the MOVE action. This is used to scale the continuous action inputs.
    render_mode: str | None = None
    window_width: int = 1200
    window_height: int = 1200
    ball_pickup_hitbox: dict[str, float] = field(
        default_factory=lambda: {
            'dist_threshold': 40, # cm
            'angle_threshold': math.radians(180), 
        }
    ) # robot can pick up ball if satisfy distance and angle thesholds.
    goal_action_hitbox: dict[str, float] = field(
        default_factory=lambda: {
            'dist_threshold': ROBOT_SIZE,
            'angle_threshold': math.radians(180),
        }
    ) # robot can score if satisfy distance and angle thesholds to the scoring position of the goal.


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
            self.renderer = EnvRenderer(
                window_width=self.config.window_width,
                window_height=self.config.window_height,
                pickup_dist_threshold=self.config.ball_pickup_hitbox["dist_threshold"],
                pickup_angle_threshold_deg=math.degrees(self.config.ball_pickup_hitbox["angle_threshold"]),
                goal_dist_threshold=self.config.goal_action_hitbox["dist_threshold"],
                goal_angle_threshold_deg=math.degrees(self.config.goal_action_hitbox["angle_threshold"]),
            )

    def _build_world(self) -> None:
        self.space = create_space()
        self.field_dict = {
            "wall": Wall(self.space),
            "CGU": CenterGoalUpper(self.space),
            "CGL": CenterGoalLower(self.space),
            "LG1": LongGoalOne(self.space),
            "LG2": LongGoalTwo(self.space),
            "legs": [
                LongGoalOneLegOne(self.space),
                LongGoalOneLegTwo(self.space),
                LongGoalTwoLegOne(self.space),
                LongGoalTwoLegTwo(self.space),
            ],
            "red_robot": Robot(self.space, position=INITIAL_RED_ROBOT_POSITION_CM, angle=math.radians(INITAL_RED_ROBOT_ANGLE_DEG), team="red"),
            "blue_robot": Robot(self.space, position=INITIAL_BLUE_ROBOT_POSITION_CM, angle=math.radians(INITAL_BLUE_ROBOT_ANGLE_DEG), team="blue"),
            "balls": [
                Ball(self.space, position=position, color="red")
                for position in INITIAL_BALL_POSITIONS_CM["red"]
            ]
            + [
                Ball(self.space, position=position, color="blue")
                for position in INITIAL_BALL_POSITIONS_CM["blue"]
            ],
        }

    def reset(self) -> Dict[str, Any]:
        self._build_world()
        self.sim_time_s = 0.0
        self.decision_elapsed_s = 0.0
        self.render_elapsed_s = 0.0
        self.done = False
        if self.renderer is not None:
            self.render()
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
        
        self._update_ball_relative_states()
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
                pass
            elif dis_act == 3: # PICKUP GROUND
                robot.pickup_ground()
            elif dis_act == 4: # SCORE
                robot.score_goal()
            elif dis_act == 5: # BLOCK
                robot.block_goal()
        
        self.update_world()
        for player in ["red", "blue"]:
            robot = self.field_dict[f"{player}_robot"]
            if action[player]["discrete"] == 3:
                robot.clear_pickup_attempt()
            if action[player]["discrete"] in (4, 5):
                robot.clear_goal_action_attempt()
        self._update_ball_relative_states()
        legal_actions = self.get_legal_actions()
        return {'legal_actions': legal_actions} # TODO: also return observation and reward

    def _update_ball_relative_states(self):
        min_r, min_b = float("inf"), float("inf")
        dist_threshold = self.config.ball_pickup_hitbox["dist_threshold"]
        angle_threshold = self.config.ball_pickup_hitbox["angle_threshold"]
        red_robot = self.field_dict["red_robot"]
        blue_robot = self.field_dict["blue_robot"]
        red_robot._pickup_ball = None
        blue_robot._pickup_ball = None
        for ball in self.field_dict["balls"]:
            rb_r = ball.update_relative_to_robot(red_robot, "robot_red")
            rb_b = ball.update_relative_to_robot(blue_robot, "robot_blue")
            # check which ball is within pickup range here so no re-calculation is needed in get_legal_actions.
            if ball.state != Ball.STATE_GROUND:
                continue
            if rb_r['distance'] <= dist_threshold and abs(rb_r['delta_theta']) <= angle_threshold and ball.color_key == "red" and rb_r['distance'] < min_r:
                red_robot._pickup_ball = ball
                min_r = min(min_r, rb_r['distance'])
            if rb_b['distance'] <= dist_threshold and abs(rb_b['delta_theta']) <= angle_threshold and ball.color_key == "blue" and rb_b['distance'] < min_b:
                blue_robot._pickup_ball = ball
                min_b = min(min_b, rb_b['distance'])

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

    def _find_nearest_goal_action_target(self, robot, opponent_robot, goals, require_goal_accept: bool):
        goal_dist_threshold = self.config.goal_action_hitbox["dist_threshold"]
        goal_angle_threshold = self.config.goal_action_hitbox["angle_threshold"]
        blocker_pos = (opponent_robot.body.position.x, opponent_robot.body.position.y)

        nearest_distance = float("inf")
        nearest_target = None

        for goal in goals:
            if require_goal_accept:
                side_position_pairs = zip(goal.score_side, goal.scoring_position)
            else:
                side_position_pairs = enumerate(goal.scoring_position)

            for side_index, scoring_position in side_position_pairs:
                dx = scoring_position[0] - robot.body.position.x
                dy = scoring_position[1] - robot.body.position.y
                target_distance = math.hypot(dx, dy)
                target_heading = math.atan2(
                    goal.body.position.y - scoring_position[1],
                    goal.body.position.x - scoring_position[0],
                )
                heading_error = (target_heading - robot.body.angle + math.pi) % (2 * math.pi) - math.pi
                within_goal_hitbox = (
                    target_distance <= goal_dist_threshold and abs(heading_error) <= goal_angle_threshold
                )
                if not within_goal_hitbox:
                    continue

                if require_goal_accept and not goal.can_accept(entry_side=side_index, blocker_pos=blocker_pos):
                    continue

                if target_distance < nearest_distance:
                    nearest_distance = target_distance
                    nearest_target = (scoring_position, target_heading, goal, side_index, opponent_robot)

        return nearest_target
        
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

            score_target = None
            block_target = None

            if len(robot.inventory) > 0:
                score_target = self._find_nearest_goal_action_target(
                    robot=robot,
                    opponent_robot=opponent_robot,
                    goals=goals,
                    require_goal_accept=True,
                )

            block_target = self._find_nearest_goal_action_target(
                robot=robot,
                opponent_robot=opponent_robot,
                goals=goals,
                require_goal_accept=False,
            )

            can_score = score_target is not None
            can_block = block_target is not None

            discrete_mask = [True, True, True, robot._pickup_ball is not None, can_score, can_block]
            legal_actions[player] = {
                "discrete_mask": discrete_mask,
            }
            robot._goal_score_target = score_target
            robot._goal_block_target = block_target

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
    env = VexEnv(Config(render_mode='human', engine_hz=60.0, env_hz=5.0, max_duration_s=120.0))
    env.reset()
    start_time = time.time()
    try:
        while not env.done:
            action = {}
            legal_actions = env.get_legal_actions()
            for player in ["red", "blue"]:
                theta = random.uniform(-math.pi, math.pi)
                possible_actions = [i for i in range(1, 6) if legal_actions[player]["discrete_mask"][i]]
                if 3 in possible_actions:
                    da = 3
                elif 4 in possible_actions:
                    da = 4
                else:
                    da = 1
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
    finally:
        env.close()
    end_time = time.time()
    print(f"Episode finished in {end_time - start_time:.2f} seconds.")
    print(f"Total actions taken: {env.total_actions}")

