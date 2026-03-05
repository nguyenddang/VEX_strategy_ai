import pymunk
import math
import random


FIELD_WIDTH_INCH = 140.43
FIELD_HEIGHT_INCH = 140.41
INCH_TO_CM = 2.54
FIELD_WIDTH = 356.6922  # cm, interior (excluding walls)
FIELD_HEIGHT = 356.6414  # cm, interior (excluding walls)
ROBOT_SIZE = 38.1 # cm
ROBOT_MASS = 50 # kg, need to verify with Dylan
MAX_VELOCITY = 100 # cm/s
ROBOT_ANGULAR_VELOCITY = math.radians(360) # rad/s
WALL_THICKNESS = 3.2258
BALL_MASS_KG = 0.5 # 500 grams
BALL_RADIUS = 4.1021 # cm
BALL_LINEAR_DAMPING = 0.95
BALL_ANGULAR_DAMPING = 0.99
BALL_STOP_SPEED_CM_S = 1.0
BALL_STOP_ANGULAR_DEG_S = 5.0
CENTER_GOAL_LOWER_LENGTH = 57.399 # cm
CENTER_GOAL_LOWER_WIDTH = 10.531 # cm
CENTER_GOAL_UPPER_LENGTH = 57.399 # cm
CENTER_GOAL_UPPER_WIDTH = 14.053 # cm
CENTER_GOAL_POSITION = (FIELD_WIDTH / 2, FIELD_HEIGHT / 2)
LOADERS_POSITION = [(2.58, 23.44), (2.58, 116.97), (137.87, 116.97), (137.87, 23.44)] # inch  
LOADERS_WIDTH = 11.811 # cm 
LOADERS_LENGTH = 10.5918 # cm  
LONG_GOAL_LENGTH = 123.925 # cm
LONG_GOAL_WIDTH = 14.053 # cm
LONG_GOAL_1_POSITION = (178.3588, 297.1038)
LONG_GOAL_2_POSITION = (178.3588, 59.5376)
LONG_GOAL_LEG_SIZE = 10.16 # cm
LONG_GOAL_LEG_X1 = 120.0468
LONG_GOAL_LEG_X2 = 236.6708
PICKUP_GROUND_COMPLETION_DIST = ROBOT_SIZE / 2 + BALL_RADIUS * (2.0 / 3.0) # "intake" overlap with 1/3 of ball.
PICKUP_GROUND_APPROACH_DIST = ROBOT_SIZE / 2 + BALL_RADIUS * 1.5 # standoff distance for pickup approach.
ROBOT_ARRIVAL_EPSILON = 2.0 # cm
ROBOT_HEADING_EPSILON = math.radians(5.0) # radians
GOAL_SLOT_DEBUG = "1"
INITIAL_RED_ROBOT_POSITION = (30, 70.2) # in inch 
INITIAL_BLUE_ROBOT_POSITION = (110, 70.2) # in inch
INITAL_RED_ROBOT_ANGLE_DEG = 0.0
INITAL_BLUE_ROBOT_ANGLE_DEG = 180.0

CATEGORY_ROBOT = 0b0001
CATEGORY_BALL_RED = 0b0010
CATEGORY_BALL_BLUE = 0b0100
CATEGORY_BALL = CATEGORY_BALL_RED | CATEGORY_BALL_BLUE
CATEGORY_GOAL_UPPER = 0b1000

INITAL_BALL_POSITIONS = {
    'red': [(46.65, 46.63), (46.65, 49.88), (49.88, 46.64), (46.65, 90.53), (46.65, 93.78), (49.88, 93.78), (115.76, 1.61), (118.99, 1.61), (115.76, 138.8), 
            (118.99, 138.8), (126.42, 65.26), (126.41, 68.59), (126.42, 71.82), (126.42, 75.05), (68.6, 23.44), (65.37, 23.4), (65.37, 116.97), (68.6, 116.97)],
    'blue': []
} # in inch. Need to convert to cm when creating balls
l_temp = ['r', 'r', 'r', 'b', 'b', 'b']
INITIAL_LOADERS_BALLS = [l_temp, l_temp, list(reversed(l_temp)), list(reversed(l_temp))] # initial ball color arrangement in loaders. 
# blue is flipped of red. 
MID_POINT_X = 70.215 # inch
MID_POINT_Y = 70.205 # inch
for x, y in INITAL_BALL_POSITIONS['red']:
    INITAL_BALL_POSITIONS['blue'].append((2 * MID_POINT_X - x, 2 * MID_POINT_Y - y))
for blue_ball in INITAL_BALL_POSITIONS['blue']:
    assert blue_ball in INITAL_BALL_POSITIONS['blue'], "Blue ball positions should be flipped version of red ball positions."
def _inches_to_cm_point(point):
    return (point[0] * INCH_TO_CM, point[1] * INCH_TO_CM)

INITIAL_RED_ROBOT_POSITION_CM = _inches_to_cm_point(INITIAL_RED_ROBOT_POSITION)
INITIAL_BLUE_ROBOT_POSITION_CM = _inches_to_cm_point(INITIAL_BLUE_ROBOT_POSITION)
INITIAL_BALL_POSITIONS_CM = {
    "red": [_inches_to_cm_point(position) for position in INITAL_BALL_POSITIONS["red"]],
    "blue": [_inches_to_cm_point(position) for position in INITAL_BALL_POSITIONS["blue"]],
}
LOADERS_POSITION_CM = [_inches_to_cm_point(position) for position in LOADERS_POSITION]
LOADER_1_POSITION = LOADERS_POSITION_CM[0]
LOADER_2_POSITION = LOADERS_POSITION_CM[1]
LOADER_3_POSITION = LOADERS_POSITION_CM[2]
LOADER_4_POSITION = LOADERS_POSITION_CM[3]

def create_space():
    space = pymunk.Space()
    space.gravity = (0, 0)
    space.collision_slop = 0.0
    return space


class Wall:
    def __init__(self, space, thickness=WALL_THICKNESS):
        self.body = pymunk.Body(body_type=pymunk.Body.STATIC)
        self.shapes = []
        half_t = thickness / 2

        segments = [
            ((-half_t, -half_t), (FIELD_WIDTH + half_t, -half_t)),
            ((FIELD_WIDTH + half_t, -half_t), (FIELD_WIDTH + half_t, FIELD_HEIGHT + half_t)),
            ((FIELD_WIDTH + half_t, FIELD_HEIGHT + half_t), (-half_t, FIELD_HEIGHT + half_t)),
            ((-half_t, FIELD_HEIGHT + half_t), (-half_t, -half_t)),
        ] # basically start and end points of the 4 walls.

        for start, end in segments:
            segment = pymunk.Segment(self.body, start, end, thickness / 2)
            segment.friction = 1.0 # might need to tune friction and elasticity.
            segment.elasticity = 0.0
            self.shapes.append(segment)
        space.add(self.body, *self.shapes)


class Robot:
    def __init__(self, space, position=(FIELD_WIDTH / 4, FIELD_HEIGHT / 4), angle=0, colour="red", capacity=10):
        """Main robot class. Handles:
        - physics body and shape
        - basic motion control toward a target pose
        - pickup and scoring state machines and logic 
        """
        assert colour in ("red", "blue"), "Colour must be 'red' or 'blue'."
        self.body = pymunk.Body(
            mass=ROBOT_MASS,
            moment=pymunk.moment_for_box(ROBOT_MASS, (ROBOT_SIZE, ROBOT_SIZE)),
            body_type=pymunk.Body.DYNAMIC,
        )
        self.space = space
        self.body.position = position
        self.body.angle = angle
        self.colour = colour

        self.shape = pymunk.Poly.create_box(self.body, size=(ROBOT_SIZE, ROBOT_SIZE))
        self.shape.friction = 1.0
        self.shape.elasticity = 0.0
        self.shape.filter = pymunk.ShapeFilter(categories=CATEGORY_ROBOT, mask=pymunk.ShapeFilter.ALL_MASKS())
        space.add(self.body, self.shape)
        
        self.move_target_pos = None # store target position
        self.move_target_angle = None # store target angle in radians
        self.inventory = [
            Ball(space=space, 
                 position=position, 
                 colour=colour, 
                 state='robot_red' if colour == 'red' else 'robot_blue', 
                 add_sim=False) 
            for _ in range(2)]  # list of balls currently held by the robot, used for scoring. Balls that are in inventory has the same position as robot.
        
        self.capacity = capacity # maximum number of balls the robot can hold. reinforced by env.py
        self._pickup_ball = None # potential ball to be picked up. Only one ball can be picked up at a time.
        self._pickup_phase = None # 'align' or 'charge' or None.
        self._building_score_target = None # target for scoring attempt, tuple of (target_pos, target_heading, goal, entry_side, opponent_robot).
        self._building_block_target = None # target for blocking attempt, tuple of (target_pos, target_heading, goal, entry_side, opponent_robot).
        self._building_loader_target = None # target for loader pickup attempt, tuple of (target_pos, target_heading, loader).
        self._building_action_phase = None # 'line_up', 'charge', or None.
        self._building_action_mode = None # 'score' or 'block' or 'loader' or None.        
        
    def _set_ball_ghost(self, ball, ghost: bool):
        """Toggle whether a ball collides with robots.

        When ghost=True, a pickup target ball can pass through either
        robot while pickup logic is executing (aka ghost). This reduce collision deadlocks a bit. 
        """
        if ball is None:
            return
        if ghost:
            # Remove robot bit from the mask: keep collisions with everything else.
            ball.shape.filter = pymunk.ShapeFilter(
                categories=ball.shape.filter.categories,
                mask=pymunk.ShapeFilter.ALL_MASKS() ^ CATEGORY_ROBOT,
            )
        else:
            # restore collision
            ball.shape.filter = pymunk.ShapeFilter(
                categories=ball.shape.filter.categories,
                mask=pymunk.ShapeFilter.ALL_MASKS(),
            )

    def _normalize_angle(self, angle: float) -> float:
        """Wrap angle to [-pi, pi] to avoid wrap-around discontinuities. Input should be radians.

        E.g: from +179° to -179° should be a -2° correction, not -358°.
        """
        return (angle + math.pi) % (2 * math.pi) - math.pi

    def set_motion_target(self, target_pos, target_angle):
        """Set normal navigation target (position + heading) and cancel other modes. 
        Called in env.py under action MOVE.
        """
        self._pickup_ball = None
        self._pickup_phase = None
        self._building_score_target = None
        self._building_block_target = None
        self._building_loader_target = None
        self._building_action_phase = None
        self._building_action_mode = None
        # make sure target is within bound. avoid weird issues.
        margin = ROBOT_SIZE / 2
        clamped_x = min(max(target_pos[0], margin), FIELD_WIDTH - margin)
        clamped_y = min(max(target_pos[1], margin), FIELD_HEIGHT - margin)
        self.move_target_pos = (clamped_x, clamped_y)
        self.move_target_angle = self._normalize_angle(target_angle)

    def pickup_ground(self):
        """Start pickup behaviour toward current '_pickup_ball'.
        Called in env.py under action PICKUP_GROUND.
        """
        ball = self._pickup_ball
        self.move_target_pos = None
        self.move_target_angle = None
        self._set_ball_ghost(ball, True) # set this ball to ghost so it can pass through robots during pickup.
        self._pickup_phase = "align"

    def clear_action_attempt(self):
        """End any in-progress action attempt and restore default action state."""
        self._set_ball_ghost(self._pickup_ball, False)
        self._pickup_ball = None
        self._pickup_phase = None
        self._building_action_phase = None
        self._building_score_target = None
        self._building_block_target = None
        self._building_loader_target = None
        self._building_action_mode = None

    def score_goal(self):
        """Start scoring behaviour using current `_building_score_target`.
        """
        self._pickup_ball = None
        self._pickup_phase = None
        self.move_target_pos = None
        self.move_target_angle = None
        self._building_action_phase = "line_up"
        self._building_action_mode = "score"
        
    def block_goal(self):
        """Start blocking behavior using current `_building_block_target`.
        """
        self._pickup_ball = None
        self._pickup_phase = None
        self.move_target_pos = None
        self.move_target_angle = None
        self._building_action_phase = "line_up"
        self._building_action_mode = "block"

    def pickup_loader(self):
        """Start loader pickup behavior using current `_building_loader_target`."""
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
        """Low-level controller: set linear + angular velocity toward target pose.
        Always move toward target pose at constant translation velocity while applying constant angular velocity. 
        """
        dx = target_pos[0] - self.body.position[0]
        dy = target_pos[1] - self.body.position[1]
        distance = math.hypot(dx, dy)
        self.body.velocity = ((dx / distance) * MAX_VELOCITY, (dy / distance) * MAX_VELOCITY) if distance > ROBOT_ARRIVAL_EPSILON else (0, 0)
        angle_error = self._normalize_angle(target_angle - self.body.angle)
        sign = 1 if angle_error > 0 else -1
        self.body.angular_velocity = ROBOT_ANGULAR_VELOCITY * sign if abs(angle_error) > ROBOT_HEADING_EPSILON else 0
        return distance, abs(angle_error)

    def _update_pickup_ground(self):
        """ Pickup state machine (`align` -> `charge` -> success).

        - `align`: move to a standoff point behind the ball while facing it.
        - `charge`: assumed 'aligned'. Charge toward the ball to pick it up.
        - success: when distance between ball and robot crosses the completion threshold AND
          intake heading is aligned, remove ball from physics and append it
          to `inventory`.
        """
        assert self._pickup_ball is not None and self._pickup_phase in ("align", "charge"), "Pickup phase should be 'align' or 'charge' when updating pickup."
        ball_x = self._pickup_ball.body.position.x
        ball_y = self._pickup_ball.body.position.y
        dx = ball_x - self.body.position.x
        dy = ball_y - self.body.position.y
        delta_ball = math.hypot(dx, dy) # how far away from ball
        unit_x = dx / delta_ball if delta_ball > 1e-8 else math.cos(self.body.angle)
        unit_y = dy / delta_ball if delta_ball > 1e-8 else math.sin(self.body.angle)
        standoff_target_angle = math.atan2(dy, dx)
        standoff_target_pos = (ball_x - unit_x * PICKUP_GROUND_APPROACH_DIST, ball_y - unit_y * PICKUP_GROUND_APPROACH_DIST)
        delta_ball_angle = abs(self._normalize_angle(standoff_target_angle - self.body.angle))

        if delta_ball <= PICKUP_GROUND_COMPLETION_DIST + ROBOT_ARRIVAL_EPSILON and delta_ball_angle <= ROBOT_HEADING_EPSILON:
            # pickup successful: stop robot, remove ball from space, add ball to inventory.
            picked_ball = self._pickup_ball
            self.body.velocity = (0, 0)
            self.body.angular_velocity = 0
            picked_ball.state = "robot_red" if self.colour == "red" else "robot_blue"
            self._set_ball_ghost(picked_ball, False)
            self.space.remove(picked_ball.shape, picked_ball.body) # remove from sim.
            self.inventory.append(picked_ball)
            self._pickup_ball = None
            self._pickup_phase = None
            return True
        # aligning phase.
        if self._pickup_phase == "align":
            delta_align_dist, delta_align_angle = self._apply_motion(standoff_target_pos, standoff_target_angle)
            if delta_align_dist <= ROBOT_ARRIVAL_EPSILON and delta_align_angle <= ROBOT_HEADING_EPSILON:
                self._pickup_phase = "charge"
            return False
        if delta_ball > PICKUP_GROUND_APPROACH_DIST + ROBOT_ARRIVAL_EPSILON or delta_ball_angle > ROBOT_HEADING_EPSILON:
            # if too far go back to aligning. (this can happen if opponent bumped us out of the way)
            self._pickup_phase = "align"
            self._apply_motion(standoff_target_pos, standoff_target_angle)
            return False
        self._apply_motion((ball_x, ball_y), standoff_target_angle)
        return False

    def _update_building_action(self):
        """Building-action state machine (`line_up` -> `charge` -> success).

        Used for SCORE, BLOCK, and PICKUP LOADERS modes.

        - `line_up`: move onto the centerline and rotate toward target heading.
        - `charge`: drive to target point while maintaining heading.
        - success: requires arriving at target pose (distance + heading + lateral alignment).
        """
        active_target = self._get_active_building_action_target()
        assert active_target is not None, "_update_building_action assumes a legal active target."

        goal = None
        entry_side = None
        loader = None

        if self._building_action_mode == "loader":
            building_target_pos, building_target_angle, loader = active_target
        else:
            building_target_pos, building_target_angle, goal, entry_side, opponent_robot = active_target
        dx = building_target_pos[0] - self.body.position.x
        dy = building_target_pos[1] - self.body.position.y
        delta_target = math.hypot(dx, dy)
        delta_target_angle = abs(self._normalize_angle(building_target_angle - self.body.angle))
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
            delta_target <= ROBOT_ARRIVAL_EPSILON
            and abs(delta_line_lateral) <= ROBOT_ARRIVAL_EPSILON
            and delta_target_angle <= ROBOT_HEADING_EPSILON
        ):
            self.body.velocity = (0, 0)
            self.body.angular_velocity = 0
            self._building_action_phase = None
            self._building_score_target = None
            self._building_block_target = None

            if self._building_action_mode == "score":
                # at position to score, try to push ball into goal.
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
                loader.pickup_loader(
                    robot=self
                )
                return True
            self._building_action_mode = None
            return True
        if self._building_action_phase == "line_up":
            self._apply_motion(lineup_target_pos, building_target_angle)
            if abs(delta_line_lateral) <= ROBOT_ARRIVAL_EPSILON and delta_target_angle <= ROBOT_HEADING_EPSILON:
                self._building_action_phase = "charge"
            return False
        if abs(delta_line_lateral) > ROBOT_ARRIVAL_EPSILON or delta_target_angle > ROBOT_HEADING_EPSILON:
            self._building_action_phase = "line_up"
            self._apply_motion(lineup_target_pos, building_target_angle)
            return False
        self._apply_motion(building_target_pos, building_target_angle)
        return False

    def stop(self):
        """Clear move targets and zero body velocities.
        """
        self.move_target_pos = None
        self.move_target_angle = None
        self.body.velocity = (0, 0)
        self.body.angular_velocity = 0

    def update(self, dt):
        """Per-tick robot update.

        Priority order:
        1) action behaviour
        2) normal move target tracking
        3) idle stop

        Note: While pickup is active, MOVE targets are ignored until pickup
        finishes or is canceled.
        """
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
        if distance <= ROBOT_ARRIVAL_EPSILON and angle_error <= ROBOT_HEADING_EPSILON:
            self.stop()
        
class Ball:
    def __init__(self, space, position, colour="red", state="ground", add_sim=True):
        ball_colour = colour.lower()
        assert ball_colour in ("red", "blue"), "Colour must be 'red' or 'blue'."
        moment = pymunk.moment_for_circle(BALL_MASS_KG, 0, BALL_RADIUS)
        self.body = pymunk.Body(mass=BALL_MASS_KG, moment=moment, body_type=pymunk.Body.DYNAMIC)
        self.body.position = position
        self.body.velocity_func = self._apply_rolling_resistance
        self.shape = pymunk.Circle(self.body, BALL_RADIUS)
        self.shape.friction = 0.7
        self.shape.elasticity = 0.1
        ball_category = CATEGORY_BALL_RED if ball_colour == "red" else CATEGORY_BALL_BLUE
        self.shape.filter = pymunk.ShapeFilter(categories=ball_category, mask=pymunk.ShapeFilter.ALL_MASKS())
        self.colour = ball_colour
        self.state = state
        self.relative_robot = {}
        if add_sim:
            space.add(self.body, self.shape)

    def _apply_rolling_resistance(self, body, gravity, damping, dt):
        """Resistance for more 'realistic' ball physics.
        """
        pymunk.Body.update_velocity(body, gravity, BALL_LINEAR_DAMPING, dt)
        body.angular_velocity *= BALL_ANGULAR_DAMPING ** dt
        stop_angular_rad = math.radians(BALL_STOP_ANGULAR_DEG_S)
        if body.velocity.length < BALL_STOP_SPEED_CM_S:
            body.velocity = (0, 0)
        if abs(body.angular_velocity) < stop_angular_rad:
            body.angular_velocity = 0
            
    def update_relative_to_robot(self, robot_red, robot_blue):
        """Update relative metric to both robot and return it.
        ALso check if ball can be picked up.

        Metric includes:
        - dx, dy: ball position relative to robot position in world frame.
        - distance: center distance between ball and robot.
        - delta_theta: angle difference between robot heading and ball relative position.
        """
        
        for robot, robot_key in [(robot_red, "robot_red"), (robot_blue, "robot_blue")]:
            dx = self.body.position.x - robot.body.position.x
            dy = self.body.position.y - robot.body.position.y
            relative_distance = math.hypot(dx, dy)
            delta_theta = robot._normalize_angle(math.atan2(dy, dx) - robot.body.angle)
            self.relative_robot[robot_key] = {
                "dx": dx,
                "dy": dy,
                "distance": relative_distance,
                "delta_theta": delta_theta,
            }   
        return self.relative_robot["robot_red"], self.relative_robot["robot_blue"]



class Goal:
    def __init__(self, space, position, length, width, angle_deg, goal_key, capacity=0, shape_category=None, mask=None):
        assert capacity >= 0, "Goal capacity cannot be negative."
        self.body = pymunk.Body(body_type=pymunk.Body.STATIC)
        self.body.position = position
        self.body.angle = math.radians(angle_deg)
        self.shape = pymunk.Poly.create_box(self.body, size=(length, width))
        if shape_category is not None and mask is not None:
            self.shape.filter = pymunk.ShapeFilter(categories=shape_category, mask=mask)
        self.shape.friction = 1.0
        self.shape.elasticity = 0.0
        self.length = length
        self.width = width
        self.capacity = capacity
        self.scored_balls = [None] * capacity if capacity > 0 else []
        self.goal_key = goal_key
        self.score_side = [0, 1]
        self.scoring_position = self._create_scoring_position()
        space.add(self.body, self.shape)

    def _slot_world_position(self, slot_index: int):
        """Get xy pos from slot idx
        """
        slot_count = max(1, self.capacity)
        axis_x = math.cos(self.body.angle)
        axis_y = math.sin(self.body.angle)
        step = self.length / (slot_count + 1)
        start_offset = -self.length / 2 + step
        offset = start_offset + slot_index * step
        return (
            self.body.position.x + axis_x * offset,
            self.body.position.y + axis_y * offset,
        )

    def _debug_slot_signature(self):
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

    def can_accept(self, entry_side: int, blocker_pos):
        filled_slots = sum(ball is not None for ball in self.scored_balls)
        if filled_slots < self.capacity:
            return True
        output_side = 1 if entry_side == 0 else 0
        output_pos = self.scoring_position[output_side]
        dx = blocker_pos[0] - output_pos[0]
        dy = blocker_pos[1] - output_pos[1]
        heading_error = math.atan2(dy, dx) - self.body.angle
        heading_error = (heading_error + math.pi) % (2 * math.pi) - math.pi
        is_blocking = abs(heading_error) <= ROBOT_HEADING_EPSILON and math.hypot(dx, dy) <= ROBOT_ARRIVAL_EPSILON
        return not is_blocking
    

    def score_goal(self, ball, entry_side: int):
        side = 0 if entry_side == 0 else 1
        goal_space = self.body.space

        before_signature = self._debug_slot_signature() if GOAL_SLOT_DEBUG else None

        # a new ball enters at the entry side and pushes existing balls toward the opposite side.
        ejected_ball = None
        def insert_from_left():
            nonlocal ejected_ball
            empty_slot = None
            for idx in range(self.capacity):
                if self.scored_balls[idx] is None:
                    empty_slot = idx
                    break
            if empty_slot is None:
                # no empty slot, the last ball will be ejected.
                ejected_ball = self.scored_balls[self.capacity - 1]
                for idx in range(self.capacity - 1, 0, -1):
                    self.scored_balls[idx] = self.scored_balls[idx - 1]
            else:
                # shift balls to the right until the empty slot by 1.
                for idx in range(empty_slot, 0, -1):
                    self.scored_balls[idx] = self.scored_balls[idx - 1]

            self.scored_balls[0] = ball
            assert self.goal_key is not None, "Goal should have a goal_key attribute for ball state assignment."
            ball.state = self.goal_key

        if side == 0:
            insert_from_left()
        else:
            # mirror slots so right-entry is equivalent to left-entry.
            self.scored_balls.reverse()
            insert_from_left()
            self.scored_balls.reverse()

        for slot_idx, slot_ball in enumerate(self.scored_balls):
            if slot_ball is None:
                continue
            slot_pos = self._slot_world_position(slot_idx)
            slot_ball.body.position = slot_pos
            slot_ball.body.velocity = (0, 0)
            slot_ball.body.angular_velocity = 0

        if ejected_ball is not None:
            ejected_ball.state = "ground"
            output_side = 1 if side == 0 else 0
            output_pos = self.scoring_position[output_side]
            out_dx = output_pos[0] - self.body.position.x
            out_dy = output_pos[1] - self.body.position.y
            out_dist = math.hypot(out_dx, out_dy)
            out_x = out_dx / out_dist if out_dist > 1e-8 else math.cos(self.body.angle)
            out_y = out_dy / out_dist if out_dist > 1e-8 else math.sin(self.body.angle)
            eject_distance = self.length / 2 + BALL_RADIUS * 1.25
            ejected_ball.body.position = (self.body.position.x + out_x * eject_distance, self.body.position.y + out_y * eject_distance)
            ejected_ball.body.velocity = (0, 0)
            ejected_ball.body.angular_velocity = 0
            goal_space.add(ejected_ball.body, ejected_ball.shape)

        if GOAL_SLOT_DEBUG:
            goal_name = getattr(self, "goal_key", self.__class__.__name__)
            entry_name = "left" if side == 0 else "right"
            incoming_colour = getattr(ball, "colour", "?")
            ejected_colour = getattr(ejected_ball, "colour", None) if ejected_ball is not None else None
            after_signature = self._debug_slot_signature()
            print(
                f"[GOAL_DEBUG] goal={goal_name} entry={entry_name} in={incoming_colour} "
                f"before={before_signature} after={after_signature} ejected={ejected_colour}"
            )

        return True
    
    def _create_scoring_position(self):
        half_length = self.length / 2
        half_robot = ROBOT_SIZE / 2
        clearance = half_length + half_robot + 2
        cos = math.cos(self.body.angle)
        if self.goal_key == 'center_lower':
            return [
                (
                    self.body.position.x - cos * clearance,
                    self.body.position.y - cos * clearance,
                ),
                (
                    self.body.position.x + cos * clearance,
                    self.body.position.y + cos * clearance,
                ),
            ]
        elif self.goal_key == 'center_upper':
            return [
                (
                    self.body.position.x - cos * clearance,
                    self.body.position.y + cos * clearance,
                ),
                (
                    self.body.position.x + cos * clearance,
                    self.body.position.y - cos * clearance,
                ),
            ]
        else:
            return [
                (
                    self.body.position.x - cos * clearance,
                    self.body.position.y
                ),
                (
                    self.body.position.x + cos * clearance,
                    self.body.position.y
                )
            ]
            
    def get_game_score(self):
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

        if self.goal_key.startswith('long'):
            red_control_count = 0
            blue_control_count = 0
            for control_index in (7, 8, 9):
                control_ball = self.scored_balls[control_index]
                if control_ball is None:
                    continue
                if control_ball.colour == "red":
                    red_control_count += 1
                elif control_ball.colour == "blue":
                    blue_control_count += 1

            if red_control_count > blue_control_count:
                red_score += 10
            elif blue_control_count > red_control_count:
                blue_score += 10
        elif self.goal_key == 'center_lower':
            if red_ball_count > blue_ball_count:
                red_score += 6
            elif blue_ball_count > red_ball_count:
                blue_score += 6
        elif self.goal_key == 'center_upper':
            if red_ball_count > blue_ball_count:
                red_score += 8
            elif blue_ball_count > red_ball_count:
                blue_score += 8

        return red_score, blue_score


class Loader:
    def __init__(self, space, position, goal_key: str, initial_ball_codes=None, manager=None):
        assert initial_ball_codes is not None, 'Loader should be initialized with initial_ball_codes for pre-population.'
        self.body = pymunk.Body(body_type=pymunk.Body.STATIC)
        self.body.position = position
        self.body.angle = 0.0
        self.shape = pymunk.Poly.create_box(self.body, size=(LOADERS_WIDTH, LOADERS_LENGTH))
        self.shape.friction = 1.0
        self.shape.elasticity = 0.0
        self.goal_key = goal_key
        self.capacity = 6
        self.scored_balls = [None] * self.capacity

        half_length = LOADERS_LENGTH / 2 
        half_robot = ROBOT_SIZE / 2
        shift = half_length + half_robot + 2
        shift = shift if position[0] < MID_POINT_X else -shift
        self.loading_position = (position[0] + shift, position[1])

        space.add(self.body, self.shape)
        if initial_ball_codes is not None:
            self.prepopulate_balls(initial_ball_codes)
        self.manager = manager # reference to Loader_Manager for refilling after pickup.

    def prepopulate_balls(self, color_codes):
        """ Pre-populate balls before match starts.
        """
        for slot_idx, colour_code in enumerate(color_codes):
            ball_colour = "red" if colour_code == "r" else "blue"
            ball = Ball(
                self.body.space,
                position=(self.body.position.x, self.body.position.y),
                colour=ball_colour,
                state=self.goal_key,
                add_sim=False,
            )
            ball.state = self.goal_key
            ball.body.position = (self.body.position.x, self.body.position.y)
            ball.body.velocity = (0, 0)
            ball.body.angular_velocity = 0
            self.scored_balls[slot_idx] = ball

    def pickup_loader(self, robot):
        picked_ball = self.scored_balls[0]
        assert picked_ball is not None, "pickup_loader assumes legal actions and requires an available loader ball."
        for idx in range(self.capacity - 1):
            self.scored_balls[idx] = self.scored_balls[idx + 1]
        self.scored_balls[self.capacity - 1] = None
        picked_ball.body.velocity = (0, 0)
        picked_ball.body.angular_velocity = 0
        if picked_ball.colour == robot.colour:
            picked_ball.state = f"robot_{robot.colour}"
            picked_ball.body.position = robot.body.position
            robot.inventory.append(picked_ball)
        else:
            # robot loaded ball is different color,immediately drop on the ground.
            drop_distance = ROBOT_SIZE / 2 + BALL_RADIUS * 1.25
            behind_x = -math.cos(robot.body.angle)
            behind_y = -math.sin(robot.body.angle)
            picked_ball.state = "ground"
            picked_ball.body.position = (
                robot.body.position.x + behind_x * drop_distance,
                robot.body.position.y + behind_y * drop_distance,
            )
            picked_ball.body.velocity = (0, 0)
            picked_ball.body.angular_velocity = 0
            self.body.space.add(picked_ball.body, picked_ball.shape)
        # request manager to refill 
        self.manager.fill(self)
class Loader_Manager:
    
    def __init__(self, space, colour):
        self.left_to_load = 12
        self.inventory = [Ball(
            space=space,
            position=(-1, -1),
            colour=colour,
            state='N/A',
            add_sim=False
        ) for _ in range(self.left_to_load)]
        self.colour = colour
        
    def fill(self, loader):
        if self.left_to_load <= 0:
            return
        fill_ball = self.inventory.pop(0)
        fill_ball.state = loader.goal_key
        fill_ball.body.position = loader.body.position
        fill_ball.body.velocity = (0, 0)
        fill_ball.body.angular_velocity = 0
        loader.scored_balls[-1] = fill_ball
        self.left_to_load -= 1
        print(f"[LOADER_MANAGER] Filled loader {loader.goal_key} with a {fill_ball.colour} ball. Balls left to load: {self.left_to_load}")
        


class Leg:
    def __init__(self, space, position):
        self.body = pymunk.Body(body_type=pymunk.Body.STATIC)
        self.body.position = position
        self.body.angle = 0.0
        self.shape = pymunk.Poly.create_box(self.body, size=(LONG_GOAL_LEG_SIZE, LONG_GOAL_LEG_SIZE))
        self.shape.friction = 1.0
        self.shape.elasticity = 0.0
        space.add(self.body, self.shape)


def step_space(space, dt):
    substeps = 4
    sub_dt = dt / substeps
    for _ in range(substeps):
        space.step(sub_dt)
        
