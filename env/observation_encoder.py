import time 
import torch
from env.type import Field
from env.config import EnvConfig
import math
from typing import Dict, Any
class ObservationEncoder:
    def __init__(self, env_config: EnvConfig, engine_config: Dict[str, Any]):
        self.env_config = env_config
        self.engine_config = engine_config
        self.max_actions = int(self.env_config.max_duration_s * self.env_config.inference_hz)
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
        """
        field_dict = field.to_field_dict()
        observations = {
            'robot_red': {
                'core': None,
                'balls': None,
            },
            'robot_blue': {
                'core': None,
                'balls': None,
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
                time_norm, score_diff_norm, bottom_long_control, upper_long_control, cgl_majority, cgu_majority,
                x_norm, y_norm, cos_theta, sin_theta, opp_x_norm, opp_y_norm, opp_cos_theta, opp_sin_theta,
                score_norm, opp_score_norm, robot_inventory_norm
            ], dtype=torch.float32)
            observations[key]['core'] = core_obs
            ball_obs = self._get_balls_obs(field_dict, robot_key=key)
            observations[key]['balls'] = ball_obs
        return observations

    def _long_control_bool(self, field_dict: Dict[str, Any], robot_key) -> bool: 
        """Return 2 boolean indicating control zone capture for LG1, and LG2 of robot_key.
        Since observation is red-canonical, for robot_blue, LG1 would be LG2 and LG2 would be LG1. 
        Boolean return indicate control for bottom long goal (LG1 in red-canonical) and upper long goal (LG2 in red-canonical) respectively.
        """
        if robot_key == "robot_red":
            bottom_control = field_dict["LG1"].has_control_zone(robot_key)
            upper_control = field_dict["LG2"].has_control_zone(robot_key)
        else:
            bottom_control = field_dict["LG2"].has_control_zone(robot_key)
            upper_control = field_dict["LG1"].has_control_zone(robot_key)
        return bottom_control, upper_control
    
    def center_majority_bool(self, field_dict: Dict[str, Any], robot_key) -> bool:
        """Return 2 boolean indicating majority control for center lower and center upper for robot_key. 
        Center goals are not affected by red-canonical. 
        Boolean return indicate majority for center lower and center upper respectively.
        """
        cgl_majority = field_dict["CGL"].has_majority(robot_key)
        cgu_majority = field_dict["CGU"].has_majority(robot_key)
        return cgl_majority, cgu_majority
    
    def _get_position_norm(self, field_dict: Dict[str, Any], robot_key: str):
        """Returns x_norm, y_norm, cos(theta), sin(theta) for robot_key and opponent robot in red-canonical frame.          
        Function ensures blue robot observations are mirrored into red-canonical frame. 
        """
        curr_robot = field_dict[robot_key]
        x, y = curr_robot.body.position
        theta = curr_robot.body.angle
        x_norm, y_norm = x / self.engine_config['field']['width'], y / self.engine_config['field']['height']
        cos_theta, sin_theta = math.cos(theta), math.sin(theta)
        
        opp_robot = field_dict["robot_blue"] if robot_key == "robot_red" else field_dict["robot_red"]
        opp_x, opp_y = opp_robot.body.position
        opp_theta = opp_robot.body.angle
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
            - x_norm, y_norm: position
            - dx_norm, dy_norm: relative position to robot normalized by field dimension
            - dist_norm: normalized distance to robot
            - heading_err_sin, heading_err_cos: sin and cos of heading error to robot
            - one_hot_state (19): return 1 number. 
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
        opp_robot = field_dict["robot_blue"] if robot_key == "robot_red" else field_dict["robot_red"]
        balls_obs_p1 = []
        ball_obs_p2_idx = []
        x_curr_robot, y_curr_robot = field_dict[robot_key].body.position
        for i, ball in enumerate(field_dict["balls"]):
            relative = ball.relative_stats[robot_key]
            dx, dy, dist, delta_theta = relative["dx"], relative["dy"], relative["distance"], relative["delta_theta"]
            dx_norm, dy_norm = dx / self.engine_config['field']['width'], dy / self.engine_config['field']['height']
            x, y = x_curr_robot + dx, y_curr_robot + dy
            x_norm, y_norm = x / self.engine_config['field']['width'], y / self.engine_config['field']['height']
            heading_err_sin, heading_err_cos = math.sin(delta_theta), math.cos(delta_theta)
            dist_norm = dist / math.hypot(self.engine_config['field']['width'], self.engine_config['field']['height'])
            ball_state = ball.state
            ball_colour = ball.colour if robot_key == 'robot_red' else ("red" if ball.colour == "blue" else "blue")
            
            if ball_state == 'long_1':
                ball_state = 'long_2' if robot_key == 'robot_blue' else 'long_1'
            elif ball_state == 'long_2':
                ball_state = 'long_1' if robot_key == 'robot_blue' else 'long_2'
            
            if ball_state == robot_key:
                one_hot_idx = self.ball_state_to_one_hot["in_possession"][ball_colour]
            elif ball_state == opp_robot.key or ball_state == "N/A":
                one_hot_idx = self.ball_state_to_one_hot["N/A"]
            else:
                one_hot_idx = self.ball_state_to_one_hot[ball_state][ball_colour]
            ball_obs_p2_idx.append(one_hot_idx)
            if ball_state == 'N/A' or ball_state == opp_robot.key:
                x_norm, y_norm, dx_norm, dy_norm, dist_norm, heading_err_sin, heading_err_cos = (-2.0,) * 7 
            elif robot_key == "robot_blue":
                x_norm, y_norm = 1 - x_norm, 1 - y_norm
                dx_norm, dy_norm = -dx_norm, -dy_norm
            balls_obs_p1.append([x_norm, y_norm, dx_norm, dy_norm, dist_norm, heading_err_sin, heading_err_cos])
        
        ball_obs_p1 = torch.tensor(balls_obs_p1, dtype=torch.float32)
        ball_obs_p2_idx = torch.tensor(ball_obs_p2_idx, dtype=torch.long)
        ball_obs_p2_one_hot = torch.nn.functional.one_hot(ball_obs_p2_idx, num_classes=21)
        ball_obs = torch.cat([ball_obs_p1, ball_obs_p2_one_hot], dim=1)
        return ball_obs
        
