import time
from typing import Any, Dict, List
import math

from env.utils import build_world, update_world, step_space, get_match_score, reset_world
from env.legal_actions import LegalActionResolver
from config import VexConfig
from env.observation_encoder import ObservationEncoder

import torch 
class VexEnv:
    def __init__(self, config: VexConfig):
        self.main_config = config
        self.engine_config = config.engine_config
        self.observation_encoder = ObservationEncoder(config=config)
        self.legal_action_resolver = LegalActionResolver(
            goal_action_hitbox=self.main_config.goal_action_hitbox,
            loader_pickup_hitbox=self.main_config.loader_pickup_hitbox,
            ball_pickup_hitbox=self.main_config.ball_pickup_hitbox,
        )
        self.n_engine_updates = self.main_config.n_engine_updates
        self.n_render_updates = self.main_config.n_render_updates
        self.max_actions = self.main_config.max_actions
        self.engine_update_dt = 1.0 / self.main_config.engine_hz
        self.renderer = None
        self.time_temp = {}
        self.field = None

        if self.main_config.render_mode == "human":
            from env.renderer import EnvRenderer
            self.renderer = EnvRenderer(config=config)
        

    def _build_world(self) -> None:
        if self.field is None:
            print("Building world...")
            self.space, self.field = build_world(self.engine_config)
        else:
            self.field = reset_world(self.space, self.field, self.engine_config)
        # self.space, self.field = build_world(self.engine_config)
        self._update_match_score()
        for robot in [self.field.robot_red, self.field.robot_blue]:
            for ball in robot.inventory:
                ball.body.position = robot.body.position
        self._update_cache_pose()

    def reset(self) -> Dict[str, Any]:
        self._build_world()
        done = False
        if self.renderer is not None:
            self.render()
        legal_actions = self.legal_action_resolver.get_legal_actions(field=self.field)
        self._update_match_score()
        observations = self._get_observations()
        return {
            'done': done,
            'legal_actions': legal_actions,
            'observations': observations,
            'rewards': {'robot_red': 0.0, 'robot_blue': 0.0},
            'timestep': 0,
            'score': {'robot_red': self.field.red_score, 'robot_blue': self.field.blue_score},
        }

    def step(self, action: Dict[str, List[int]]) -> Dict[str, Any]:
        """Step env

        Args:
            action : Action dict for both red and blue.
            Default expected frame is canonical-per-player. Under this frame,
            blue MOVE x/y bins are mirrored into world frame so both policies can
            act in a shared red-canonical action space.

            Required keys per player:
            - discrete: primary action type index [0..5]
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
            
            action example:
            {
                'robot_red': [1 (MOVE), 3 (x-bin), 4 (y-bin), 5 (theta-bin)],
                'robot_blue': [1 (MOVE), 7 (x-bin), 9 (y-bin), 2 (theta-bin)], # x,y,theta bin are red-canonical. Env will mirror them into world frame for blue.
            }
        """
        self.field.actions_counter += 1
        action = self._process_policy_action(action)
        prev_red_score, prev_blue_score = self.field.red_score, self.field.blue_score
        prev_red_inven, prev_blue_inven = len(self.field.robot_red.inventory), len(self.field.robot_blue.inventory)
        for player in ["robot_red", "robot_blue"]:
            dis_act = action[player][0]
            robot = self.field.robot_red if player == "robot_red" else self.field.robot_blue
            if dis_act == 0: # NO-OP
                pass
            elif dis_act == 1: # MOVE
                x_idx = action[player][1]
                y_idx = action[player][2]
                theta_idx = action[player][3]

                delta_x = -self.main_config.max_offset + (2.0 * self.main_config.max_offset * x_idx) / (self.main_config.N - 1)
                delta_y = -self.main_config.max_offset + (2.0 * self.main_config.max_offset * y_idx) / (self.main_config.N - 1)
                delta_theta = -math.pi + (2.0 * math.pi * (theta_idx + 0.5)) / self.main_config.K
                target_x = robot.body.position.x + delta_x
                target_y = robot.body.position.y + delta_y
                target_theta = robot.body.angle + delta_theta
                robot.set_motion_target((target_x, target_y), target_theta)
            elif dis_act == 2: # PICKUP LOADERS
                robot.pickup_loader()
                # print(f"{player} attempts to pickup loader at timestep {self.field.actions_counter}")
            elif dis_act == 3: # PICKUP GROUND
                robot.pickup_ground()
                # print(f"{player} attempts to pickup ground ball at timestep {self.field.actions_counter}")
            elif dis_act == 4: # SCORE
                robot.score_goal()
                # print(f"{player} attempts to score at timestep {self.field.actions_counter}")
            elif dis_act == 5: # BLOCK
                robot.block_goal()
                # print(f"{player} attempts to block at timestep {self.field.actions_counter}")
        
        self._update_world()
        for player in ["red", "blue"]:
            robot = self.field.robot_red if player == "red" else self.field.robot_blue
            robot.clear_action_attempt()
        legal_actions = self.legal_action_resolver.get_legal_actions(field=self.field)
        done = self.field.actions_counter >= self.max_actions
        observations = self._get_observations()
        post_red_score, post_blue_score = self.field.red_score, self.field.blue_score
        post_red_inven, post_blue_inven = len(self.field.robot_red.inventory), len(self.field.robot_blue.inventory)
        reward_red, reward_blue = self._get_rewards(
            prev_red_score, prev_blue_score, prev_red_inven, prev_blue_inven,
            post_red_score, post_blue_score, post_red_inven, post_blue_inven,
            done = done
        )
        return {
            'legal_actions': legal_actions,
            'observations': observations,
            'done': done,
            'rewards': {'robot_red': reward_red, 'robot_blue': reward_blue},
            'timestep': self.field.actions_counter,
            'score': {'robot_red': post_red_score, 'robot_blue': post_blue_score},
        }

    def _update_world(self):
        update_world(
            field=self.field,
            n_engine_updates=self.n_engine_updates,
            engine_update_dt=self.engine_update_dt,
            n_render_updates=self.n_render_updates,
            step_space_fn=step_space,
            render_fn=self.render if self.renderer is not None else None,
            realtime=self.renderer is not None and self.main_config.realtime_render,
        )
        self._update_match_score()
        for robot in [self.field.robot_red, self.field.robot_blue]:
            for ball in robot.inventory:
                ball.body.position = robot.body.position
        self._update_cache_pose()
        
    def _update_match_score(self):
        red_score, blue_score = get_match_score(self.field)
        self.field.red_score = red_score
        self.field.blue_score = blue_score
        
    def _get_observations(self) -> Dict[str, Any]:
        observations = self.observation_encoder.encode(self.field)
        return observations
    
    def _get_rewards(
        self,
        prev_red_score: int,
        prev_blue_score: int,
        prev_red_inven: int,
        prev_blue_inven: int,
        post_red_score: int,
        post_blue_score: int,
        post_red_inven: int,
        post_blue_inven: int,
        done: bool,
    ):
        reward_red, reward_blue = 0, 0
        reward_red += (post_red_score - prev_red_score) * 0.1
        reward_red += 0.1 if post_red_inven > prev_red_inven else 0 # incentivize pickup
        reward_blue += (post_blue_score - prev_blue_score) * 0.1
        reward_blue += 0.1 if post_blue_inven > prev_blue_inven else 0 # incentivize pickup
        if done:
            if post_red_score > post_blue_score:
                reward_red += 2.0
                score_diff = post_red_score - post_blue_score
                # bonus for winning by larger margin
                reward_red += 0.01 * score_diff
            elif post_blue_score > post_red_score:
                reward_blue += 2.0
                score_diff = post_blue_score - post_red_score
                # bonus for winning by larger margin
                reward_blue += 0.01 * score_diff
        return reward_red, reward_blue 
    
    def _process_policy_action(self, action: Dict[str, Any]) -> Dict[str, Any]:
        new_blue_act = []
        new_blue_act.append(self.main_config.N - 1 - action["robot_blue"][1])
        new_blue_act.append(self.main_config.N - 1 - action["robot_blue"][2])
        new_blue_act.append(action["robot_blue"][3]) # theta does not need to be rotated, as rotation is symmetric
        new_action = {
            "robot_red": action["robot_red"],
            "robot_blue": [action["robot_blue"][0]] + new_blue_act
        }
        return new_action
    
    def _update_cache_pose(self):
        for ball in self.field.balls:
            ball._update_cache_pose()
        for robot in [self.field.robot_red, self.field.robot_blue]:
            robot._update_cache_pose()
        
    def render(self):
        if self.renderer is None:
            return None
        return self.renderer.render(self.field)

    def close(self):
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None
