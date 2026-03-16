import pymunk
import math
from typing import Dict, Any, Optional, Tuple, List
import random 
from .utils import normalize_angle

class Wall:
    def __init__(
        self, 
        space: pymunk.Space, 
        wall_config:Dict[str, Any], 
        field_config:Dict[str, Any]
        ):
        self.body = pymunk.Body(body_type=pymunk.Body.STATIC)
        self.shapes = []
        half_t = wall_config['thickness'] / 2
        field_width, field_height = field_config['width'], field_config['height']

        segments = [
            ((-half_t, -half_t), (field_width + half_t, -half_t)),
            ((field_width + half_t, -half_t), (field_width + half_t, field_height + half_t)),
            ((field_width + half_t, field_height + half_t), (-half_t, field_height + half_t)),
            ((-half_t, field_height + half_t), (-half_t, -half_t)),
        ]

        for start, end in segments:
            segment = pymunk.Segment(self.body, start, end, half_t)
            segment.friction = 1.0
            segment.elasticity = 0.0
            self.shapes.append(segment)
        space.add(self.body, *self.shapes)

class Leg:
    def __init__(
        self, space: pymunk.Space, 
        leg_config:Dict[str, Any], 
        key: str,   
        ):
        self.body = pymunk.Body(body_type=pymunk.Body.STATIC)
        self.body.position = leg_config['position']
        self.body.angle = 0.0
        self.shape = pymunk.Poly.create_box(self.body, size=(leg_config['size'], leg_config['size']))
        self.shape.friction = 1.0
        self.shape.elasticity = 0.0
        self.key = key
        space.add(self.body, self.shape)
        
class Ball:
    def __init__(
        self, 
        space: pymunk.Space, 
        ball_config:Dict[str, Any], 
        colour: str, 
        state: str, 
        add_sim: bool=True, 
        position: Tuple[float, float]=(-2.0, -2.0),
        loader_level: float=-1,
        ):
        
        self.colour = colour
        self.config = ball_config
        self.state = state
        self.inital_state = state
        self.add_sim = add_sim
        self.inital_position = position
        # init sim
        moment = pymunk.moment_for_circle(ball_config['mass'], 0, ball_config['radius'])
        self.body = pymunk.Body(mass=ball_config['mass'], moment=moment, body_type=pymunk.Body.DYNAMIC)
        self.body.position = position
        self.body.velocity_func = self._apply_rolling_resistance
        self.cache_pose = {'position': (self.body.position.x, self.body.position.y), 'angle': self.body.angle}
        self.shape = pymunk.Circle(self.body, ball_config['radius'])
        self.shape.friction = ball_config['friction']
        self.shape.elasticity = ball_config['elasticity']
        self.shape.filter = pymunk.ShapeFilter(categories=2 if colour == "red" else 4, mask=pymunk.ShapeFilter.ALL_MASKS())
        self.relative_stats = {}
        if add_sim:
            space.add(self.body, self.shape)
        self.stop_angular_speed = math.radians(self.config['stop_angular_speed'])
        self.loader_level = loader_level

    def _apply_rolling_resistance(self, body:pymunk.Body, gravity, damping, dt):
        """ Custom damping. to make things look good:)
        """
        if self.state != "ground":
            pymunk.Body.update_velocity(body, gravity, damping, dt) 
            return
        pymunk.Body.update_velocity(body, gravity, self.config['linear_damping'], dt)

    def update_relative_to_robot(self, robot):
        """Calculate relative distance and angle to each robot for observation.
        """
        x_robot, y_robot = robot.cache_pose['position']
        theta_robot = robot.cache_pose['angle']
        dx = self.cache_pose['position'][0] - x_robot
        dy = self.cache_pose['position'][1] - y_robot
        relative_distance = math.hypot(dx, dy)
        delta_theta = normalize_angle(math.atan2(dy, dx) - theta_robot)
        self.relative_stats[robot.key] = {
            "dx": dx,
            "dy": dy,
            "distance": relative_distance,
            "delta_theta": delta_theta,
        }
        return self.relative_stats[robot.key]
    
    def _update_cache_pose(self):
        """Cache position to reduce repeated pymunk body.position call.
        """
        body = self.body
        position, angle = body.position, body.angle
        self.cache_pose['position'] = (position.x, position.y)
        self.cache_pose['angle'] = angle

class Goal:
    def __init__(
        self, 
        space: pymunk.Space, 
        goal_config:Dict[str, Any],
        blocking_config:Dict[str, Any],
        ball_config:Dict[str, Any],
        robot_config:Dict[str, Any],
        key: str,
        debug: bool=False,
        ):
        # basic attr
        position = goal_config['position']
        self.height = goal_config['height']
        self.width = goal_config['width']
        self.key = key
        self.capacity = goal_config['capacity']
        self.angle = goal_config['angle']
        self.blocking_config = blocking_config
        self.ball_config = ball_config
        self.robot_config = robot_config
        self.debug = debug # print debug info for each scoring action if True
        # init sim
        self.body = pymunk.Body(body_type=pymunk.Body.STATIC)
        self.body.position = position
        self.body.angle = math.radians(self.angle)
        self.shape = pymunk.Poly.create_box(self.body, size=(self.width, self.height))
        if self.key != 'center_lower':
            self.shape.filter = pymunk.ShapeFilter(categories=8, mask=1)
        self.shape.friction = 1.0
        self.shape.elasticity = 0.0
        self.cache_pose = {'position': (self.body.position.x, self.body.position.y), 'angle': self.body.angle}
        # keep track of scored balls
        self.scored_balls: List[Optional[Ball]] = [None] * self.capacity
        self.score_side = [0, 1]
        self.scoring_position = self._create_scoring_position()
        self.slot_position = [self._slot_world_position(i) for i in range(self.capacity)]
        space.add(self.body, self.shape)
        self.relative_stats = {}

    def _slot_world_position(self, slot_index: int):
        """Calculate xy of each ball in the goal
        """
        step = self.width / (self.capacity + 1)
        start_offset = -self.width / 2 + step
        offset = start_offset + slot_index * step
        return (
            self.cache_pose['position'][0] + math.cos(self.cache_pose['angle']) * offset,
            self.cache_pose['position'][1] + math.sin(self.cache_pose['angle']) * offset,
        )
        
    def _update_relative_to_robot(self, robot) -> List[Dict[str, float]]:
        """Calculate relative distance and angle to the robot from scoring position for observation.
        """
        x_robot, y_robot = robot.cache_pose['position']
        theta_robot = robot.cache_pose['angle']
        temp = []
        for score_pos in self.scoring_position:
            dx = score_pos[0] - x_robot
            dy = score_pos[1] - y_robot
            relative_distance = math.hypot(dx, dy)
            delta_theta = normalize_angle(math.atan2(dy, dx) - theta_robot)
            temp.append({
                "dx": dx,
                "dy": dy,
                "distance": relative_distance,
                "delta_theta": delta_theta,
            })
        self.relative_stats[robot.key] = temp
        return self.relative_stats[robot.key]

    def _debug_slot_signature(self):
        """Debugging prints
        """
        signature = []
        for slot_ball in self.scored_balls:
            if slot_ball is None:
                signature.append("0")
                continue
            slot_ball_colour = getattr(slot_ball, "colour", None)
            if slot_ball_colour == "red":
                signature.append("R")
            elif slot_ball_colour == "blue":
                signature.append("B")
            else:
                signature.append("?")
        return signature

    def can_accept(self, entry_side: int, blocker_pos: Tuple[float, float]):
        """Given scoring side and blocker position, determine if goal can accept new ball.
        """
        filled_slots = sum(ball is not None for ball in self.scored_balls)
        if filled_slots < self.capacity:
            # space available, can accept regardless of blocker
            return True
        output_side = 1 if entry_side == 0 else 0
        output_pos = self.scoring_position[output_side]
        dx = blocker_pos[0] - output_pos[0]
        dy = blocker_pos[1] - output_pos[1]
        heading_error = normalize_angle(math.atan2(dy, dx) - self.cache_pose['angle'])
        is_blocking = abs(heading_error) <= math.radians(self.blocking_config['angle_threshold']) and math.hypot(dx, dy) <= self.blocking_config['distance_threshold']
        return not is_blocking

    def score_goal(self, ball: Ball, entry_side: int):
        """ Score ball into goal. Eject ball to other side if needed. 
        Function assumes scoring action is legal.
        """
        goal_space = self.body.space
        before_signature = self._debug_slot_signature() if self.debug else None
        ejected_ball: Optional[Ball] = None
        def insert_from_left() -> Optional[Ball]:
            empty_slot = None
            # try to find empty slot and save index
            for idx in range(self.capacity):
                if self.scored_balls[idx] is None:
                    empty_slot = idx
                    break
            loop_range = range(empty_slot, 0, -1) if empty_slot is not None else range(self.capacity - 1, 0, -1)
            ejected_ball = self.scored_balls[self.capacity - 1] if empty_slot is None else None
            
            for idx in loop_range:
                self.scored_balls[idx] = self.scored_balls[idx - 1]
                self.scored_balls[idx].body.position = self.slot_position[idx]
                self.scored_balls[idx].body.velocity = (0, 0)
                self.scored_balls[idx].body.angular_velocity = 0
            self.scored_balls[0] = ball
            self.scored_balls[0].body.position = self.slot_position[0]
            ball.state = self.key
            return ejected_ball
        
        # main scoring
        if entry_side == 0:
            ejected_ball = insert_from_left()
        else:
            # flip the list so we can reuse same insertion logic.
            self.scored_balls.reverse()
            self.slot_position.reverse()
            ejected_ball = insert_from_left()
            self.scored_balls.reverse()
            self.slot_position.reverse()
        # handle ejected ball if exists
        if ejected_ball is not None:
            ejected_ball.state = "ground"
            output_side = 1 if entry_side == 0 else 0
            output_pos = self.scoring_position[output_side]
            out_dx = output_pos[0] - self.cache_pose['position'][0]
            out_dy = output_pos[1] - self.cache_pose['position'][1]
            out_dist = math.hypot(out_dx, out_dy)
            out_x = out_dx  / out_dist if out_dist > 1e-8 else math.cos(self.cache_pose['angle'])
            out_y = out_dy  / out_dist if out_dist > 1e-8 else math.sin(self.cache_pose['angle'])
            eject_distance = self.width / 2 + self.ball_config['radius'] * 1.25 
            ejected_ball.body.position = (
                self.cache_pose['position'][0] + (out_x) * (eject_distance + random.uniform(4, 16)),
                self.cache_pose['position'][1] + (out_y + random.uniform(-0.1, 0.1)) * (eject_distance + random.uniform(4, 16)),
            )
            ejected_ball.body.velocity = (out_x * 100, out_y * 100)
            ejected_ball.body.angular_velocity = 0
            goal_space.add(ejected_ball.body, ejected_ball.shape)

        if self.debug:
            entry_name = "left" if entry_side == 0 else "right"
            after_signature = self._debug_slot_signature()
            print(
                f"[GOAL_DEBUG] goal={self.key} entry={entry_name} in={ball.colour} "
                f"before={before_signature} after={after_signature} ejected={ejected_ball.colour if ejected_ball is not None else None}"
            )

    def _create_scoring_position(self):
        """Calculate xy coord robot need to be at to score. 
        """
        half_length = self.width / 2
        half_robot = self.robot_config['size'] / 2
        clearance = half_length + half_robot + 2
        cos = math.cos(self.cache_pose['angle'])
        if self.key == "center_lower":
            return [
                (
                    self.cache_pose['position'][0] - cos * clearance,
                    self.cache_pose['position'][1] - cos * clearance,
                ),
                (
                    self.cache_pose['position'][0] + cos * clearance,
                    self.cache_pose['position'][1] + cos * clearance,
                ),
            ]
        elif self.key == "center_upper":
            return [
                (
                    self.cache_pose['position'][0] - cos * clearance,
                    self.cache_pose['position'][1] + cos * clearance,
                ),
                (
                    self.cache_pose['position'][0] + cos * clearance,
                    self.cache_pose['position'][1] - cos * clearance,
                ),
            ]
        else:
            return [
                (
                    self.cache_pose['position'][0] - cos * clearance,
                    self.cache_pose['position'][1],
                ),
                (
                    self.cache_pose['position'][0] + cos * clearance,
                    self.cache_pose['position'][1],
                ),
            ]
    
    def has_control_zone(self, robot_key: str):
        """Check Long goal control zone. 
        """
        assert self.key in ["long_1", "long_2"], "Only long goals have control zones"
        colour = 'red' if robot_key == "robot_red" else 'blue'
        n_red, n_blue = 0, 0
        for ball in self.scored_balls[7:10]:
            if ball is None:
                continue
            if ball.colour == "red":
                n_red += 1
            elif ball.colour == "blue":
                n_blue += 1
        return n_red > n_blue if colour == "red" else n_blue > n_red
    
    def has_majority(self, robot_key: str):
        """Check center goal majority. 
        """
        assert self.key in ["center_lower", "center_upper"], "Only center goals have majority"
        colour = 'red' if robot_key == "robot_red" else 'blue'
        n_red, n_blue = 0, 0
        for ball in self.scored_balls:
            if ball is None:
                continue
            if ball.colour == "red":
                n_red += 1
            elif ball.colour == "blue":
                n_blue += 1
        return n_red > n_blue if colour == "red" else n_blue > n_red

    def get_game_score(self):
        """Get game score from this goal.
        - Each ball +3 points.
        - Long goal control bonus: +10 points for team with more balls in long goal slots (slots 7,8,9)
        - Center lower majority bonus: +6 points
        - Center upper majority bonus: +8 
        """
        red_ball_count = 0
        blue_ball_count = 0
        for scored_ball in self.scored_balls:
            if scored_ball is None:
                continue
            if scored_ball.colour == "red":
                red_ball_count += 1
            elif scored_ball.colour == "blue":
                blue_ball_count += 1
        red_score = red_ball_count * 3
        blue_score = blue_ball_count * 3

        # bonus logic
        if self.key.startswith("long"):
            red_control = self.has_control_zone("robot_red")
            blue_control = self.has_control_zone("robot_blue")
            if red_control:
                red_score += 10
            elif blue_control:
                blue_score += 10
        elif self.key == "center_lower":
            if self.has_majority("robot_red"):
                red_score += 6
            elif self.has_majority("robot_blue"):
                blue_score += 6
        elif self.key == "center_upper":
            if self.has_majority("robot_red"):
                red_score += 8
            elif self.has_majority("robot_blue"):
                blue_score += 8
        return red_score, blue_score
    
class Loader_Manager:
    """Helper class to fill loader with matchload balls.
    """
    def __init__(
        self, 
        space: pymunk.Space, 
        colour: str,
        ball_config:Dict[str, Any],
        debug: bool=False,
        ):
        self.left_to_load = 12
        self.inventory = [
            Ball(
                space=space,
                ball_config=ball_config,
                colour=colour,
                state="N/A",
                add_sim=False, # don't add to sim until loaded into loader
                position=(-2.0, -2.0), # (-2, -2) is NULL
            )
            for _ in range(self.left_to_load)
        ]
        self.colour = colour
        self.debug = debug # print debug info for each loading action if True
        

    def fill(self, loader):
        if self.left_to_load <= 0:
            return
        fill_ball = self.inventory.pop(0)
        fill_ball.state = loader.key
        fill_ball.body.position = loader.body.position
        fill_ball.body.velocity = (0, 0)
        fill_ball.body.angular_velocity = 0
        loader.scored_balls[-1] = fill_ball
        fill_ball.loader_level = 1.0
        self.left_to_load -= 1
        if self.debug:
            print(
                f"[LOADER_MANAGER] Filled loader {loader.key} with a {fill_ball.colour} ball. "
                f"Balls left to load: {self.left_to_load}"
            )
        
class Loader:
    def __init__(
        self, 
        space: pymunk.Space, 
        loader_config:Dict[str, Any],
        robot_config:Dict[str, Any],
        ball_config:Dict[str, Any],
        manager: Loader_Manager,
        key: str,
        ):
        
        position = loader_config['position']
        self.body = pymunk.Body(body_type=pymunk.Body.STATIC)
        self.body.position = position
        self.body.angle = 0.0
        self.cache_pose = {'position': (self.body.position.x, self.body.position.y), 'angle': self.body.angle}
        self.shape = pymunk.Poly.create_box(self.body, size=(loader_config['width'], loader_config['height']))
        self.shape.friction = 1.0
        self.shape.elasticity = 0.0
        
        shift = loader_config['width'] / 2 + robot_config['size'] / 2 + 2
        shift = shift if int(key[-1]) in [1, 2] else -shift
        self.loading_position = (position[0] + shift, position[1])

        space.add(self.body, self.shape)
        self.ball_config = ball_config
        self.manager = manager
        self.robot_config = robot_config
        self.loader_config = loader_config
        self.key = key
        self.capacity = loader_config['capacity']
        self.scored_balls: List[Optional[Ball]] = [None] * self.capacity
        self._prepopulate_balls(loader_config['preload'])
        self.relative_stats = {}
        
    def _update_relative_to_robot(self, robot):
        """Calculate relative distance and angle to the robot from loading position for observation.
        """
        x_robot, y_robot = robot.cache_pose['position']
        theta_robot = robot.cache_pose['angle']
        dx = self.loading_position[0] - x_robot
        dy = self.loading_position[1] - y_robot
        relative_distance = math.hypot(dx, dy)
        delta_theta = normalize_angle(math.atan2(dy, dx) - theta_robot)
        self.relative_stats[robot.key] = {
            "dx": dx,
            "dy": dy,
            "distance": relative_distance,
            "delta_theta": delta_theta,
        }
        return self.relative_stats[robot.key]

    def _prepopulate_balls(self, preload_list: List[str]):
        for slot_idx, ball_colour in enumerate(preload_list):
            ball = Ball(
                self.body.space,
                ball_config=self.ball_config,
                colour=ball_colour,
                state=self.key,
                add_sim=False,
                position=(self.cache_pose['position'][0], self.cache_pose['position'][1]),
                loader_level=slot_idx / (self.capacity - 1)
            )
            ball.body.velocity = (0, 0)
            ball.body.angular_velocity = 0
            self.scored_balls[slot_idx] = ball

    def pickup_loader(self, robot):
        picked_ball = self.scored_balls[0]
        for idx in range(self.capacity - 1):
            # shift all balls forward by one slot
            self.scored_balls[idx] = self.scored_balls[idx + 1]
            if self.scored_balls[idx] is not None:
                self.scored_balls[idx].loader_level = idx / (self.capacity - 1) 
        self.scored_balls[self.capacity - 1] = None
        picked_ball.body.velocity = (0, 0)
        picked_ball.body.angular_velocity = 0
        if picked_ball.colour == robot.key.split("_")[1]: # if ball colour matches robot colour, pick up the ball
            picked_ball.state = robot.key
            picked_ball.body.position = robot.body.position
            picked_ball.loader_level = -1.0
            robot.inventory.append(picked_ball)
        else:
            drop_distance = self.robot_config['size'] / 2 + self.ball_config['radius'] * 1.25
            behind_x = -math.cos(robot.body.angle)
            behind_y = -math.sin(robot.body.angle)
            picked_ball.state = "ground"
            picked_ball.body.position = (
                robot.body.position.x + behind_x * (drop_distance),
                robot.body.position.y + behind_y * drop_distance,
            )
            picked_ball.body.velocity = (behind_x * 100, behind_y * 100)  # Adjust the multiplier as needed
            picked_ball.body.angular_velocity = 0
            picked_ball.loader_level = -1.0
            self.body.space.add(picked_ball.body, picked_ball.shape)
        self.manager.fill(self) # trigger manager to fill loader after each pickup

