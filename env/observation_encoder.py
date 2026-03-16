from env.type import Field
from config import VexConfig

import torch

import math
from typing import Dict, Any
class ObservationEncoder:
    def __init__(self, config: VexConfig):
        self.main_config = config
        self.engine_config = config.engine_config
        self.max_actions = int(self.main_config.max_duration_s * self.main_config.inference_hz)
        self.ball_state_to_one_hot = {
            "ground": {"red": 0, "blue": 1},
            "long_1": {"red": 2, "blue": 3},
            "long_2": {"red": 4, "blue": 5},
            "center_lower": {"red": 6, "blue": 7},
            "center_upper": {"red": 8, "blue": 9},
            "loader_1": {"red": 10, "blue": 11},
            "loader_2": {"red": 12, "blue": 13},
            "loader_3": {"red": 14, "blue": 15},
            "loader_4": {"red": 16, "blue": 17},
            "in_possession": {"red": 18, "blue": 19},
            'N/A': 20,
        }
        self.time_temp = {}

    def encode(self, field: Field) -> Dict[str, torch.Tensor]:
        """Encode field state into observations for both robots.
        Observations are in red-canonical frame, so blue robot observations are mirrored into red-canonical frame.
        Observation scheme (for each robot):
            core_obs (85):
                - time_norm (1): normalized time step in the episode (0 to 1)
                - score_diff_norm (1): normalized score difference (own score - opponent score) / 166
                - bottom_long_control (3): one-hot [current_view_control, opponent_view_control, nobody_control]
                - upper_long_control (3): one-hot [current_view_control, opponent_view_control, nobody_control]
                - cgl_majority (3): one-hot [current_view_majority, opponent_view_majority, nobody_majority]
                - cgu_majority (3): one-hot [current_view_majority, opponent_view_majority, nobody_majority]
                - x_norm, y_norm, cos_theta, sin_theta (4): normalized x and y positions and heading of robot
                - opp_x_norm, opp_y_norm, opp_cos_theta, opp_sin_theta (4): normalized x and y positions and heading of opponent robot
                - score_norm, opp_score_norm (2): normalized score of robot and opponent
                - robot_inventory_norm (1): normalized inventory size of robot
                - goal/loader relative block (60): flattened (12 x 5) in this row order:
                  [LG1 side0, LG1 side1, LG2 side0, LG2 side1, CGL side0, CGL side1, CGU side0, CGU side1, LD1, LD2, LD3, LD4]
                  and feature order [dx, dy, dist, heading_err_sin, heading_err_cos].
                  For robot_blue, long goals and loaders are flipped to preserve red-canonical semantics.
            ball_obs (num_balls x 29): for each ball, see _get_balls_obs for details
        """
        field_dict = field.to_field_dict()
        observations = {
            'robot_red': {
                'core_obs': None,
                'ball_obs': None,
            },
            'robot_blue': {
                'core_obs': None,
                'ball_obs': None,
            }
        }
        red_score, blue_score = field_dict['red_score'], field_dict['blue_score']
        for key in observations.keys():
            time_norm = field.actions_counter / self.max_actions
            score_diff = red_score - blue_score if key == 'robot_red' else blue_score - red_score
            score_diff_norm = score_diff / 166.0
            bottom_long_control, upper_long_control = self._long_control_bool(field_dict, robot_key=key) 
            cgl_majority, cgu_majority = self.center_majority_bool(field_dict, robot_key=key)
            x_norm, y_norm, cos_theta, sin_theta, opp_x_norm, opp_y_norm, opp_cos_theta, opp_sin_theta = self._get_position_norm(field_dict, robot_key=key)
            score_norm = red_score / 166.0 if key == 'robot_red' else blue_score / 166.0
            opp_score_norm = blue_score / 166.0 if key == 'robot_red' else red_score / 166.0
            robot_inventory_norm = len(field_dict[key].inventory) / self.engine_config['robot']['capacity']
            core_obs = torch.tensor([
                time_norm, score_diff_norm,
                *bottom_long_control, *upper_long_control, *cgl_majority, *cgu_majority,
                x_norm, y_norm, cos_theta, sin_theta, opp_x_norm, opp_y_norm, opp_cos_theta, opp_sin_theta,
                score_norm, opp_score_norm, robot_inventory_norm
            ], dtype=torch.float32)
            goal_loader_obs = self._get_goal_loader_obs(field_dict, robot_key=key)
            core_obs = torch.cat([core_obs, goal_loader_obs], dim=0)
            observations[key]['core_obs'] = core_obs
            ball_obs = self._get_balls_obs(field_dict, robot_key=key)
            observations[key]['ball_obs'] = ball_obs
        return observations


    def _owner_one_hot(self, current_view_has_control: bool, opponent_view_has_control: bool) -> list[float]:
        """Encode control ownership as one-hot [current_view, opponent_view, nobody]."""
        if current_view_has_control:
            return [1.0, 0.0, 0.0]
        if opponent_view_has_control:
            return [0.0, 1.0, 0.0]
        return [0.0, 0.0, 1.0]

    def _long_control_bool(self, field_dict: Dict[str, Any], robot_key): 
        """Return two one-hot vectors for long-goal control in red-canonical order.

        Since observation is red-canonical, for robot_blue, LG1 would be LG2 and LG2 would be LG1. 
        Returns:
            - bottom long goal ownership one-hot [current_view, opponent_view, nobody]
            - upper long goal ownership one-hot [current_view, opponent_view, nobody]
        """
        if robot_key == "robot_red":
            bottom_goal, upper_goal, opponent_key = field_dict["long_1"], field_dict["long_2"], "robot_blue"
        else:
            bottom_goal, upper_goal, opponent_key = field_dict["long_2"], field_dict["long_1"], "robot_red"
        bottom_control = self._owner_one_hot(
            current_view_has_control=bottom_goal.has_control_zone(robot_key),
            opponent_view_has_control=bottom_goal.has_control_zone(opponent_key),
        )
        upper_control = self._owner_one_hot(
            current_view_has_control=upper_goal.has_control_zone(robot_key),
            opponent_view_has_control=upper_goal.has_control_zone(opponent_key),
        )
        return bottom_control, upper_control
    
    def center_majority_bool(self, field_dict: Dict[str, Any], robot_key):
        """Return two one-hot vectors for center-goal majority ownership.

        Center goals are not affected by red-canonical. 
        Returns:
            - center lower majority one-hot [current_view, opponent_view, nobody]
            - center upper majority one-hot [current_view, opponent_view, nobody]
        """
        opponent_key = "robot_blue" if robot_key == "robot_red" else "robot_red"
        cgl_majority = self._owner_one_hot(
            current_view_has_control=field_dict["center_lower"].has_majority(robot_key),
            opponent_view_has_control=field_dict["center_lower"].has_majority(opponent_key),
        )
        cgu_majority = self._owner_one_hot(
            current_view_has_control=field_dict["center_upper"].has_majority(robot_key),
            opponent_view_has_control=field_dict["center_upper"].has_majority(opponent_key),
        )
        return cgl_majority, cgu_majority
    
    def _get_goal_loader_obs(self, field_dict: Dict[str, Any], robot_key: str):
        """Return relative position obs for goals and loaders for robot_key. 
        
        Obs order: [LG1 side0, LG1 side1, LG2 side0, LG2 side1, CGL side0, CGL side1, CGU side0, CGU side1, LD1, LD2, LD3, LD4]
        For robot_blue, long goals and loaders are flipped to preserve red-canonical semantics.
        Each obs is [dx_norm, dy_norm, dist_norm, heading_err_sin, heading_err_cos].
        """
        ORDER_RED = ['long_1', 'long_2', 'center_lower', 'center_upper', 'loader_1', 'loader_2', 'loader_3', 'loader_4']
        ORDER_BLUE = ['long_2', 'long_1', 'center_lower', 'center_upper', 'loader_3', 'loader_4', 'loader_1', 'loader_2']
        side = [0, 1] if robot_key == "robot_red" else [1, 0]
        order = ORDER_RED if robot_key == "robot_red" else ORDER_BLUE
        raw_obs = []
        for key in order:
            if not key.startswith("loader"):
                for s in side:
                    relative = field_dict[key].relative_stats[robot_key][s]
                    dx, dy, dist, delta_theta = relative["dx"], relative["dy"], relative["distance"], relative["delta_theta"]
                    dx_norm, dy_norm = dx / self.engine_config['field']['width'], dy / self.engine_config['field']['height']
                    heading_err_sin, heading_err_cos = math.sin(delta_theta), math.cos(delta_theta)
                    dist_norm = dist / math.hypot(self.engine_config['field']['width'], self.engine_config['field']['height'])
                    if robot_key == "robot_blue":
                        dx_norm, dy_norm = -dx_norm, -dy_norm
                    raw_obs.append([dx_norm, dy_norm, dist_norm, heading_err_sin, heading_err_cos])
        for key in order:
            if key.startswith("loader"):
                relative = field_dict[key].relative_stats[robot_key]
                dx, dy, dist, delta_theta = relative["dx"], relative["dy"], relative["distance"], relative["delta_theta"]
                dx_norm, dy_norm = dx / self.engine_config['field']['width'], dy / self.engine_config['field']['height']
                heading_err_sin, heading_err_cos = math.sin(delta_theta), math.cos(delta_theta)
                dist_norm = dist / math.hypot(self.engine_config['field']['width'], self.engine_config['field']['height'])
                if robot_key == "robot_blue":
                    dx_norm, dy_norm = -dx_norm, -dy_norm
                raw_obs.append([dx_norm, dy_norm, dist_norm, heading_err_sin, heading_err_cos])
        raw_obs = torch.tensor(raw_obs, dtype=torch.float32).flatten()
        return raw_obs

    def _get_position_norm(self, field_dict: Dict[str, Any], robot_key: str):
        """Returns x_norm, y_norm, cos(theta), sin(theta) for robot_key and opponent robot in red-canonical frame.          
        Function ensures blue robot observations are mirrored into red-canonical frame. 
        """
        curr_robot = field_dict[robot_key]
        x, y = curr_robot.cache_pose['position']
        theta = curr_robot.cache_pose['angle']
        x_norm, y_norm = x / self.engine_config['field']['width'], y / self.engine_config['field']['height']
        cos_theta, sin_theta = math.cos(theta), math.sin(theta)
        
        opp_robot = field_dict["robot_blue"] if robot_key == "robot_red" else field_dict["robot_red"]
        opp_x, opp_y = opp_robot.cache_pose['position']
        opp_theta = opp_robot.cache_pose['angle']
        opp_x_norm, opp_y_norm = opp_x / self.engine_config['field']['width'], opp_y / self.engine_config['field']['height']
        opp_cos_theta, opp_sin_theta = math.cos(opp_theta), math.sin(opp_theta)
        
        if robot_key == "robot_blue":
            x_norm, y_norm = 1 - x_norm, 1 - y_norm
            cos_theta, sin_theta = -cos_theta, -sin_theta
            opp_x_norm, opp_y_norm = 1 - opp_x_norm, 1 - opp_y_norm
            opp_cos_theta, opp_sin_theta = -opp_cos_theta, -opp_sin_theta
        return x_norm, y_norm, cos_theta, sin_theta, opp_x_norm, opp_y_norm, opp_cos_theta, opp_sin_theta
    
    def _get_balls_obs(self, field_dict: Dict[str, Any], robot_key: str):
        """Returns ball observations for robot_key. 
        Ball observations (for each ball):
            - x_norm, y_norm, loader_level: position
            - dx_norm, dy_norm: relative position to robot normalized by field dimension
            - dist_norm: normalized distance to robot
            - heading_err_sin, heading_err_cos: sin and cos of heading error to robot
            - one_hot_state (21): return 1 number. 
                - 0, 1: ground red, blue
                - 2, 3: bottom long red, blue
                - 4, 5: upper long red, blue
                - 6, 7: center lower red, blue
                - 8, 9: center upper red, blue
                - 10, 11: loader 1 red, blue
                - 12, 13: loader 2 red, blue
                - 14, 15: loader 3 red, blue
                - 16, 17: loader 4 red, blue
                - 18, 19: ball in possession red, blue
                - 20: N/A (ball in loader manager or in opponent possession)
        Note:
            - For ball that are N/A, all other attributes set to -2.
            - ball obs are red-canonical
        """
        ball_map_blue = {
            'ground': 'ground',
            'long_1': 'long_2',
            'long_2': 'long_1',
            'center_lower': 'center_lower',
            'center_upper': 'center_upper',
            'loader_1': 'loader_3',
            'loader_2': 'loader_4',
            'loader_3': 'loader_1',
            'loader_4': 'loader_2',
            'N/A': 'N/A',
            'robot_red': 'robot_red',
            'robot_blue': 'robot_blue',
        }
        opp_robot = field_dict["robot_blue"] if robot_key == "robot_red" else field_dict["robot_red"]
        balls_obs_p1 = []
        ball_obs_p2_idx = []
        x_curr_robot, y_curr_robot = field_dict[robot_key].cache_pose['position']
        for i, ball in enumerate(field_dict["balls"]):
            relative = ball.relative_stats.get(robot_key)
            dx, dy, dist, delta_theta = relative["dx"], relative["dy"], relative["distance"], relative["delta_theta"]
            dx_norm, dy_norm = dx / self.engine_config['field']['width'], dy / self.engine_config['field']['height']
            x, y = x_curr_robot + dx, y_curr_robot + dy
            x_norm, y_norm = x / self.engine_config['field']['width'], y / self.engine_config['field']['height']
            loader_level = ball.loader_level
            heading_err_sin, heading_err_cos = math.sin(delta_theta), math.cos(delta_theta)
            dist_norm = dist / math.hypot(self.engine_config['field']['width'], self.engine_config['field']['height'])
            ball_state = ball.state
            ball_colour = ball.colour if robot_key == 'robot_red' else ("red" if ball.colour == "blue" else "blue")
            ball_state = ball_map_blue[ball_state] if robot_key == "robot_blue" else ball_state
            if ball_state == robot_key:
                one_hot_idx = self.ball_state_to_one_hot["in_possession"][ball_colour]
            elif ball_state == opp_robot.key or ball_state == "N/A":
                one_hot_idx = self.ball_state_to_one_hot["N/A"]
            else:
                one_hot_idx = self.ball_state_to_one_hot[ball_state][ball_colour]
            ball_obs_p2_idx.append(one_hot_idx)
            if ball_state == 'N/A' or ball_state == opp_robot.key:
                x_norm, y_norm, dx_norm, dy_norm, dist_norm, heading_err_sin, heading_err_cos = (-2.0,) * 7 
                loader_level = -1.0
            elif robot_key == "robot_blue":
                x_norm, y_norm = 1 - x_norm, 1 - y_norm
                dx_norm, dy_norm = -dx_norm, -dy_norm
            balls_obs_p1.append([x_norm, y_norm, loader_level, dx_norm, dy_norm, dist_norm, heading_err_sin, heading_err_cos])
        
        ball_obs_p1 = torch.tensor(balls_obs_p1, dtype=torch.float32)
        ball_obs_p2_idx = torch.tensor(ball_obs_p2_idx, dtype=torch.long)
        ball_obs_p2_one_hot = torch.nn.functional.one_hot(ball_obs_p2_idx, num_classes=21)
        ball_obs = torch.cat([ball_obs_p1, ball_obs_p2_one_hot], dim=1)
        return ball_obs
        
