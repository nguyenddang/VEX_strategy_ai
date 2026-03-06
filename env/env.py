import time
from typing import Any, Dict, List
import math

from env.utils import build_world, update_world, step_space, get_match_score
from env.legal_actions import LegalActionResolver
from env.config import EnvConfig
from env.observation_encoder import ObservationEncoder

class VexEnv:
    def __init__(self, env_config: EnvConfig, engine_config: Dict[str, Any]):
        self.env_config = env_config
        self.engine_config = engine_config
        self.observation_encoder = ObservationEncoder(env_config=self.env_config, engine_config=self.engine_config)
        self.legal_action_resolver = LegalActionResolver(
            goal_action_hitbox=self.env_config.goal_action_hitbox,
            loader_pickup_hitbox=self.env_config.loader_pickup_hitbox,
            ball_pickup_hitbox=self.env_config.ball_pickup_hitbox,
        )
        assert self.env_config.engine_hz >= self.env_config.inference_hz, "Engine update frequency should be higher than or equal to inference frequency"
        assert self.env_config.engine_hz % self.env_config.inference_hz == 0, "Engine update frequency should be divisible by inference frequency"
        assert self.env_config.render_hz > 0, "Render frequency must be positive"
        assert self.env_config.N % 2 == 1, "MOVE bins (N) should be odd to have a center bin"
        self.n_engine_updates = int(self.env_config.engine_hz // self.env_config.inference_hz)
        self.n_render_updates = max(1, int(self.env_config.engine_hz // self.env_config.render_hz))
        self.max_actions = int(self.env_config.max_duration_s * self.env_config.inference_hz)
        self.engine_update_dt = 1.0 / self.env_config.engine_hz
        self.renderer = None
        self.time_temp = {}

        if self.env_config.render_mode == "human":
            from env.renderer import EnvRenderer
            self.renderer = EnvRenderer(env_config=self.env_config, engine_config=self.engine_config)
        

    def _build_world(self) -> None:
        self.space, self.field = build_world(self.engine_config)

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
        }

    def step(self, action: Dict[str, List[int]]) -> Dict[str, Any]:
        """Step env

        Args:
            action : Action dict for both red and blue.
            Default expected frame is canonical-per-player. Under this frame,
            blue MOVE x/y bins are mirrored into world frame so both policies can
            act in a shared red-canonical action space.

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
            
            action example:
            {
                'robot_red': [1 (MOVE), 3 (x-bin), 4 (y-bin), 5 (theta-bin)],
                'robot_blue': [1 (MOVE), 7 (x-bin), 9 (y-bin), 2 (theta-bin)], # x,y,theta bin are red-canonical. Env will mirror them into world frame for blue.
            }
        """
        self.field.actions_counter += 1
        action = self._process_policy_action(action)
        for player in ["robot_red", "robot_blue"]:
            dis_act = action[player][0]
            robot = self.field.robot_red if player == "robot_red" else self.field.robot_blue
            if dis_act == 0: # NO-OP
                pass
            elif dis_act == 1: # MOVE
                x_idx = action[player][1]
                y_idx = action[player][2]
                theta_idx = action[player][3]

                delta_x = -self.env_config.max_offset + (2.0 * self.env_config.max_offset * x_idx) / (self.env_config.N - 1)
                delta_y = -self.env_config.max_offset + (2.0 * self.env_config.max_offset * y_idx) / (self.env_config.N - 1)
                delta_theta = -math.pi + (2.0 * math.pi * (theta_idx + 0.5)) / self.env_config.K
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
        
        self._update_world()
        for player in ["red", "blue"]:
            robot = self.field.robot_red if player == "red" else self.field.robot_blue
            robot.clear_action_attempt()
        legal_actions = self.legal_action_resolver.get_legal_actions(field=self.field)
        done = self.field.actions_counter >= self.max_actions
        observations = self._get_observations()
        return {
            'legal_actions': legal_actions,
            'observations': observations,
            'done': done,
        } # TODO: also return reward

    def _update_world(self):
        update_world(
            field=self.field,
            n_engine_updates=self.n_engine_updates,
            engine_update_dt=self.engine_update_dt,
            n_render_updates=self.n_render_updates,
            step_space_fn=step_space,
            render_fn=self.render if self.renderer is not None else None,
        )
        self._update_match_score()
        for robot in [self.field.robot_red, self.field.robot_blue]:
            for ball in robot.inventory:
                ball.body.position = robot.body.position
        self._cache_ball_positions()
        
    def _update_match_score(self):
        red_score, blue_score = get_match_score(self.field)
        self.field.red_score = red_score
        self.field.blue_score = blue_score
        
    def _get_observations(self) -> Dict[str, Any]:
        observations = self.observation_encoder.encode(self.field)
        return observations
        
    def _process_policy_action(self, action: Dict[str, Any]) -> Dict[str, Any]:
        new_blue_act = []
        new_blue_act.append(self.env_config.N - 1 - action["robot_blue"][1])
        new_blue_act.append(self.env_config.N - 1 - action["robot_blue"][2])
        new_blue_act.append(action["robot_blue"][3]) # theta does not need to be rotated, as rotation is symmetric
        new_action = {
            "robot_red": action["robot_red"],
            "robot_blue": [action["robot_blue"][0]] + new_blue_act
        }
        return new_action
    
    def _cache_ball_positions(self):
        for ball in self.field.balls:
            ball._cache_position()
        
        
    def render(self):
        if self.renderer is None:
            return None
        return self.renderer.render(self.field)

    def close(self):
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None
