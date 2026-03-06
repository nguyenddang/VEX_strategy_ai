import math
import random
import pymunk
from typing import Optional, List, Dict, Any, Tuple

from .field_component import Ball, Goal, Loader
from .utils import normalize_angle

class Robot:
    def __init__(
        self, 
        space: pymunk.Space, 
        key: str,
        robot_config: Dict[str, Any],
        field_config: Dict[str, Any],
        ball_config: Dict[str, Any],
        ):
        inital_position = robot_config[key]['initial_position']
        inital_rotation = robot_config[key]['initial_rotation']
        self.body = pymunk.Body(
            mass=robot_config['mass'],
            moment=pymunk.moment_for_box(robot_config['mass'], (robot_config['size'], robot_config['size'])),
            body_type=pymunk.Body.DYNAMIC,
        )
        self.space = space
        self.body.position = inital_position
        self.body.angle = math.radians(inital_rotation)
        self.key = key

        self.shape = pymunk.Poly.create_box(self.body, size=(robot_config['size'], robot_config['size']))
        self.shape.friction = 1.0
        self.shape.elasticity = 0.0
        self.shape.filter = pymunk.ShapeFilter(categories=1, mask=pymunk.ShapeFilter.ALL_MASKS())
        space.add(self.body, self.shape)

        self.move_target_pos = None
        self.move_target_angle = None
        self.inventory: List[Ball] = [
            Ball(
                space=space,
                ball_config=ball_config,
                position=inital_position,
                colour=key.split('_')[-1],
                state=key,
                add_sim=False,
            )
            for _ in range(2)
        ] # robot always start with 2 balls in inventory. 
        self.capacity = robot_config['capacity']
        self.robot_config = robot_config
        self.field_config = field_config
        self.max_velocity = robot_config['max_velocity']
        self.max_angular_velocity = math.radians(robot_config['max_angular_velocity'])
        self.arrival_dist_eps = robot_config['arrival_dist_eps']
        self.arrival_angle_eps = math.radians(robot_config['arrival_angle_eps'])
        self.pickup_ground_completion_dist = robot_config['pickup_ground_completion_dist']
        self.pickup_ground_approach_dist = robot_config['pickup_ground_approach_dist']
        self.cache_pose = {'position': (self.body.position.x, self.body.position.y), 'angle': self.body.angle}
        
        self._pickup_ball: Ball | None = None
        self._pickup_phase: str | None = None
        self._building_score_target: Tuple[Tuple[float, float], float, Goal, int] | None = None
        self._building_block_target: Tuple[Tuple[float, float], float, Goal, int] | None = None
        self._building_loader_target: Tuple[Tuple[float, float], float, Loader] | None = None
        self._building_action_phase: str | None = None
        self._building_action_mode: str | None = None
        
    def _update_cache_pose(self):
        body = self.body
        angle, position = body.angle, body.position
        self.cache_pose['position'] = (position.x, position.y)
        self.cache_pose['angle'] = angle


    def _set_ball_ghost(self, ball: Ball | None, ghost: bool):
        if ball is None:
            return
        if ghost:
            ball.shape.filter = pymunk.ShapeFilter(
                categories=ball.shape.filter.categories,
                mask=pymunk.ShapeFilter.ALL_MASKS() ^ 1,
            )
        else:
            ball.shape.filter = pymunk.ShapeFilter(
                categories=ball.shape.filter.categories,
                mask=pymunk.ShapeFilter.ALL_MASKS(),
            )

    def set_motion_target(self, target_pos, target_angle):
        self._pickup_ball = None
        self._pickup_phase = None
        self._building_score_target = None
        self._building_block_target = None
        self._building_loader_target = None
        self._building_action_phase = None
        self._building_action_mode = None

        margin = self.robot_config['size'] / 2
        clamped_x = min(max(target_pos[0], margin), self.field_config['width'] - margin)
        clamped_y = min(max(target_pos[1], margin), self.field_config['height'] - margin)
        self.move_target_pos = (clamped_x, clamped_y)
        self.move_target_angle = normalize_angle(target_angle)

    def pickup_ground(self):
        ball = self._pickup_ball
        self.move_target_pos = None
        self.move_target_angle = None
        self._set_ball_ghost(ball, True)
        self._pickup_phase = "align"

    def clear_action_attempt(self):
        self._set_ball_ghost(self._pickup_ball, False)
        self._pickup_ball = None
        self._pickup_phase = None
        self._building_action_phase = None
        self._building_score_target = None
        self._building_block_target = None
        self._building_loader_target = None
        self._building_action_mode = None

    def score_goal(self):
        self._pickup_ball = None
        self._pickup_phase = None
        self.move_target_pos = None
        self.move_target_angle = None
        self._building_action_phase = "line_up"
        self._building_action_mode = "score"

    def block_goal(self):
        self._pickup_ball = None
        self._pickup_phase = None
        self.move_target_pos = None
        self.move_target_angle = None
        self._building_action_phase = "line_up"
        self._building_action_mode = "block"

    def pickup_loader(self):
        self._pickup_ball = None
        self._pickup_phase = None
        self.move_target_pos = None
        self.move_target_angle = None
        self._building_action_phase = "line_up"
        self._building_action_mode = "loader"

    def _get_active_building_action_target(self):
        if self._building_action_mode == "score":
            return self._building_score_target
        if self._building_action_mode == "block":
            return self._building_block_target
        if self._building_action_mode == "loader":
            return self._building_loader_target
        return None

    def _apply_motion(self, target_pos, target_angle):
        dx = target_pos[0] - self.body.position[0]
        dy = target_pos[1] - self.body.position[1]
        distance = math.hypot(dx, dy)
        self.body.velocity = ((dx / distance) * self.max_velocity, (dy / distance) * self.max_velocity) if distance > self.arrival_dist_eps else (0, 0)
        angle_error = normalize_angle(target_angle - self.body.angle)
        sign = 1 if angle_error > 0 else -1
        self.body.angular_velocity = self.max_angular_velocity * sign if abs(angle_error) > self.arrival_angle_eps else 0
        return distance, abs(angle_error)

    def _update_pickup_ground(self):
        ball_x = self._pickup_ball.body.position.x
        ball_y = self._pickup_ball.body.position.y
        dx = ball_x - self.body.position.x
        dy = ball_y - self.body.position.y
        delta_ball = math.hypot(dx, dy)
        unit_x = dx / delta_ball if delta_ball > 1e-8 else math.cos(self.body.angle)
        unit_y = dy / delta_ball if delta_ball > 1e-8 else math.sin(self.body.angle)
        standoff_target_angle = math.atan2(dy, dx)
        standoff_target_pos = (ball_x - unit_x * self.pickup_ground_approach_dist, ball_y - unit_y * self.pickup_ground_approach_dist)
        delta_ball_angle = abs(normalize_angle(standoff_target_angle - self.body.angle))

        if delta_ball <= self.pickup_ground_completion_dist + self.arrival_dist_eps and delta_ball_angle <= self.arrival_angle_eps:
            picked_ball = self._pickup_ball
            self.body.velocity = (0, 0)
            self.body.angular_velocity = 0
            picked_ball.state = self.key
            self._set_ball_ghost(picked_ball, False)
            self.space.remove(picked_ball.shape, picked_ball.body)
            self.inventory.append(picked_ball)
            self._pickup_ball = None
            self._pickup_phase = None
            return True

        if self._pickup_phase == "align":
            delta_align_dist, delta_align_angle = self._apply_motion(standoff_target_pos, standoff_target_angle)
            if delta_align_dist <= self.arrival_dist_eps and delta_align_angle <= self.arrival_angle_eps:
                self._pickup_phase = "charge"
            return False

        if delta_ball > self.pickup_ground_approach_dist + self.arrival_dist_eps or delta_ball_angle > self.arrival_angle_eps:
            self._pickup_phase = "align"
            self._apply_motion(standoff_target_pos, standoff_target_angle)
            return False

        self._apply_motion((ball_x, ball_y), standoff_target_angle)
        return False

    def _update_building_action(self):
        active_target = self._get_active_building_action_target()
        assert active_target is not None, "_update_building_action assumes a legal active target."
        goal = None
        entry_side = None
        loader = None
        if self._building_action_mode == "loader":
            building_target_pos, building_target_angle, loader = active_target
        else:
            building_target_pos, building_target_angle, goal, entry_side = active_target
        dx = building_target_pos[0] - self.body.position.x
        dy = building_target_pos[1] - self.body.position.y
        delta_target = math.hypot(dx, dy)
        delta_target_angle = abs(normalize_angle(building_target_angle - self.body.angle))
        target_dir_x = math.cos(building_target_angle)
        target_dir_y = math.sin(building_target_angle)
        target_line_normal_x = -target_dir_y
        target_line_normal_y = target_dir_x

        delta_line_x = self.body.position.x - building_target_pos[0]
        delta_line_y = self.body.position.y - building_target_pos[1]
        delta_line_lateral = delta_line_x * target_line_normal_x + delta_line_y * target_line_normal_y
        lineup_target_pos = (
            self.body.position.x - delta_line_lateral * target_line_normal_x,
            self.body.position.y - delta_line_lateral * target_line_normal_y,
        )

        if (
            delta_target <= self.arrival_dist_eps
            and abs(delta_line_lateral) <= self.arrival_dist_eps
            and delta_target_angle <= self.arrival_angle_eps
        ):
            self.body.velocity = (0, 0)
            self.body.angular_velocity = 0
            self._building_action_phase = None
            self._building_score_target = None
            self._building_block_target = None

            if self._building_action_mode == "score":
                assert len(self.inventory) > 0, "Scoring assumes robot has at least one ball in inventory."
                selected_index = random.randrange(len(self.inventory))
                selected_ball = self.inventory[selected_index]
                goal.score_goal(
                    ball=selected_ball,
                    entry_side=entry_side,
                )
                self.inventory.pop(selected_index)
                return True

            if self._building_action_mode == "loader":
                loader.pickup_loader(robot=self)
                return True

            self._building_action_mode = None
            return True

        if self._building_action_phase == "line_up":
            self._apply_motion(lineup_target_pos, building_target_angle)
            if abs(delta_line_lateral) <= self.arrival_dist_eps and delta_target_angle <= self.arrival_angle_eps:
                self._building_action_phase = "charge"
            return False

        if abs(delta_line_lateral) > self.arrival_dist_eps or delta_target_angle > self.arrival_angle_eps:
            self._building_action_phase = "line_up"
            self._apply_motion(lineup_target_pos, building_target_angle)
            return False

        self._apply_motion(building_target_pos, building_target_angle)
        return False

    def stop(self):
        self.move_target_pos = None
        self.move_target_angle = None
        self.body.velocity = (0, 0)
        self.body.angular_velocity = 0

    def update(self, dt):
        if dt <= 0:
            return
        if self._pickup_ball is not None and self._pickup_phase is not None:
            self._update_pickup_ground()
            return
        if self._building_action_phase is not None and self._get_active_building_action_target() is not None:
            self._update_building_action()
            return
        if self.move_target_pos is None or self.move_target_angle is None:
            self.body.velocity = (0, 0)
            self.body.angular_velocity = 0
            return
        distance, angle_error = self._apply_motion(self.move_target_pos, self.move_target_angle)
        if distance <= self.arrival_dist_eps and angle_error <= self.arrival_angle_eps:
            self.stop()
