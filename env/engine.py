import pymunk
import math
import random


FIELD_WIDTH_INCH = 140.43
FIELD_HEIGHT_INCH = 140.41
INCH_TO_CM = 2.54
FIELD_WIDTH = FIELD_WIDTH_INCH * INCH_TO_CM  # cm, interior (excluding walls)
FIELD_HEIGHT = FIELD_HEIGHT_INCH * INCH_TO_CM  # cm, interior (excluding walls)
ROBOT_SIZE = 38.1 # cm
ROBOT_MASS = 50 # kg, need to verify with Dylan
MAX_VELOCITY = 100 # cm/s
ROBOT_ANGULAR_VELOCITY = math.radians(360) # rad/s
WALL_THICKNESS = 3.2258
BALL_MASS_KG = 0.5 # 500 grams
BALL_RADIUS = 4.1021 # cm
BALL_LINEAR_DAMPING = 0.99
BALL_ANGULAR_DAMPING = 0.99
BALL_STOP_SPEED_CM_S = 1.0
BALL_STOP_ANGULAR_DEG_S = 5.0
CENTER_GOAL_LOWER_LENGTH = 57.399 # cm
CENTER_GOAL_LOWER_WIDTH = 10.531 # cm
CENTER_GOAL_UPPER_LENGTH = 57.399 # cm
CENTER_GOAL_UPPER_WIDTH = 14.053 # cm
LONG_GOAL_LENGTH = 123.925 # cm
LONG_GOAL_WIDTH = 14.053 # cm
LONG_GOAL_1_POSITION = (178.3588, 297.1038)
LONG_GOAL_2_POSITION = (178.3588, 59.5376)
LONG_GOAL_LEG_SIZE = 10.16 # cm
LONG_GOAL_LEG_X1 = 120.0468
LONG_GOAL_LEG_X2 = 236.6708
PICKUP_GROUND_COMPLETION_DIST = ROBOT_SIZE / 2 + BALL_RADIUS * (2.0 / 3.0) # "intake" overlap with 1/3 of ball.
ROBOT_ARRIVAL_EPSILON = 2.0 # cm
ROBOT_HEADING_EPSILON = math.radians(5.0) # radians
GOAL_BLOCK_EPSILON_CM = ROBOT_ARRIVAL_EPSILON
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
# blue is flipped of red. 
MID_POINT_X = 70.215 # inch
MID_POINT_Y = 70.205 # inch
for x, y in INITAL_BALL_POSITIONS['red']:
    INITAL_BALL_POSITIONS['blue'].append((2 * MID_POINT_X - x, 2 * MID_POINT_Y - y))

def _inches_to_cm_point(point):
    return (point[0] * INCH_TO_CM, point[1] * INCH_TO_CM)

INITIAL_RED_ROBOT_POSITION_CM = _inches_to_cm_point(INITIAL_RED_ROBOT_POSITION)
INITIAL_BLUE_ROBOT_POSITION_CM = _inches_to_cm_point(INITIAL_BLUE_ROBOT_POSITION)
INITIAL_BALL_POSITIONS_CM = {
    "red": [_inches_to_cm_point(position) for position in INITAL_BALL_POSITIONS["red"]],
    "blue": [_inches_to_cm_point(position) for position in INITAL_BALL_POSITIONS["blue"]],
}

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
    def __init__(self, space, position=(FIELD_WIDTH / 4, FIELD_HEIGHT / 4), angle=0, team="red"):
        """Main robot class. Handles:
        - physics body and shape
        - basic motion control toward a target pose
        - pickup and scoring state machines and logic 
        """
        assert team in ("red", "blue"), "Team must be 'red' or 'blue'."
        self.body = pymunk.Body(
            mass=ROBOT_MASS,
            moment=pymunk.moment_for_box(ROBOT_MASS, (ROBOT_SIZE, ROBOT_SIZE)),
            body_type=pymunk.Body.DYNAMIC,
        )
        self.space = space
        self.body.position = position
        self.body.angle = angle
        self.team = team

        self.shape = pymunk.Poly.create_box(self.body, size=(ROBOT_SIZE, ROBOT_SIZE))
        self.shape.friction = 1.0
        self.shape.elasticity = 0.0
        self.shape.filter = pymunk.ShapeFilter(categories=CATEGORY_ROBOT, mask=pymunk.ShapeFilter.ALL_MASKS())

        self.move_target_pos = None # store target position
        self.move_target_angle = None # store target angle in radians
        self.inventory = [] # list of balls currently held by the robot, used for scoring. Balls that are in inventory has the same position as robot.
        self._pickup_ball = None # potential ball to be picked up. Only one ball can be picked up at a time.
        self._pickup_phase = None # 'align' or 'charge' or None.
        self._goal_score_target = None # target for scoring attempt, tuple of (target_pos, target_heading, goal, entry_side, opponent_robot). 
        self._goal_block_target = None # target for blocking attempt, tuple of (target_pos, target_heading, goal, entry_side, opponent_robot).
        self._goal_action_phase = None # 'line_up', 'face_goal', 'charge', or None. 
        self._goal_action_mode = None # 'score' or 'block' or None. 
        space.add(self.body, self.shape)

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
        self._goal_score_target = None
        self._goal_block_target = None
        self._goal_action_phase = None
        self._goal_action_mode = None
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

    def clear_pickup_attempt(self):
        """End current pickup attempt regardless of success or failure. R
        Restore collision state.
        """
        self._set_ball_ghost(self._pickup_ball, False)
        self._pickup_ball = None
        self._pickup_phase = None

    def score_goal(self):
        """ Start scoring behaviour using current `_goal_score_target`.
        """
        self._pickup_ball = None
        self._pickup_phase = None
        self.move_target_pos = None
        self.move_target_angle = None
        self._goal_action_phase = "line_up"
        self._goal_action_mode = "score"
        
    def block_goal(self):
        """Start blocking behavior using current `_goal_block_target`.
        """
        self._pickup_ball = None
        self._pickup_phase = None
        self.move_target_pos = None
        self.move_target_angle = None
        self._goal_action_phase = "line_up"
        self._goal_action_mode = "block"
    
    def clear_goal_action_attempt(self):
        """End goal action attempt as unsuccessful for one-step action semantics. (either SCORE_GOAL or BLOCK_GOAL)
        """
        self._goal_action_phase = None
        self._goal_score_target = None
        self._goal_block_target = None
        self._goal_action_mode = None

    def _apply_motion(self, target_pos, target_angle):
        """Low-level controller: set linear + angular velocity toward target pose.
        Always move toward target pose at constant translation velocity while applying constant angular velocity. 
        """
        dx = target_pos[0] - self.body.position[0]
        dy = target_pos[1] - self.body.position[1]
        distance = math.hypot(dx, dy)
        if distance > ROBOT_ARRIVAL_EPSILON:
            # note arrived, continue.
            self.body.velocity = (dx / distance * MAX_VELOCITY, dy / distance * MAX_VELOCITY)
        else:
            # arrvied, set velocity to 0.
            self.body.velocity = (0, 0)
        angle_error = self._normalize_angle(target_angle - self.body.angle)
        if abs(angle_error) > ROBOT_HEADING_EPSILON:
            self.body.angular_velocity = ROBOT_ANGULAR_VELOCITY if angle_error > 0 else -ROBOT_ANGULAR_VELOCITY
        else:
            self.body.angular_velocity = 0
        return distance, abs(angle_error)

    def _update_pickup_ground(self):
        """ Pickup state machine (`align` -> `charge` -> success).

        - `align`: move to a standoff point behind the ball while facing it.
        - `charge`: if heading error is large, rotate in place; otherwise drive in.
        - success: when center distance crosses the completion threshold AND
          intake heading is aligned, remove ball from physics and append it
          to `inventory`.
        """
        assert self._pickup_ball is not None and self._pickup_phase in ("align", "charge"), "Pickup phase should be 'align' or 'charge' when updating pickup."
        ball_x = self._pickup_ball.body.position.x
        ball_y = self._pickup_ball.body.position.y
        dx = ball_x - self.body.position.x
        dy = ball_y - self.body.position.y
        center_distance = math.hypot(dx, dy)

        if center_distance > 1e-8:
            unit_x = dx / center_distance
            unit_y = dy / center_distance
        else:
            # mathematically equivalent, avoid division by 0.
            unit_x = math.cos(self.body.angle)
            unit_y = math.sin(self.body.angle)

        desired_heading = math.atan2(dy, dx)
        approach_distance = ROBOT_SIZE / 2 + 1.5 * BALL_RADIUS
        align_target = (ball_x - unit_x * approach_distance, ball_y - unit_y * approach_distance)
        heading_error = abs(self._normalize_angle(desired_heading - self.body.angle))

        if (
            center_distance <= PICKUP_GROUND_COMPLETION_DIST
            and heading_error <= ROBOT_HEADING_EPSILON
        ):
            # pickup successful: stop robot, remove from space, add to inventory.
            picked_ball = self._pickup_ball
            self.body.velocity = (0, 0)
            self.body.angular_velocity = 0
            picked_ball.state = Ball.STATE_ROBOT_RED if self.team == "red" else Ball.STATE_ROBOT_BLUE
            self._set_ball_ghost(picked_ball, False)
            self.space.remove(picked_ball.shape, picked_ball.body) # remove from sim.
            self.inventory.append(picked_ball)
            self._pickup_ball = None
            self._pickup_phase = None
            return True

        # aligning phase.
        if self._pickup_phase == "align":
            align_distance, align_angle_error = self._apply_motion(align_target, desired_heading)
            if (
                (align_distance <= ROBOT_ARRIVAL_EPSILON and align_angle_error <= ROBOT_HEADING_EPSILON)
                or (center_distance <= approach_distance + ROBOT_ARRIVAL_EPSILON)
            ):
                self._pickup_phase = "charge"
            return False

        # Charge phase.
        if center_distance > approach_distance + ROBOT_ARRIVAL_EPSILON:
            self._pickup_phase = "align"
            return False

        if heading_error > ROBOT_HEADING_EPSILON:
            self._pickup_phase = "align"
            return False

        self._apply_motion((ball_x, ball_y), desired_heading)
        return False

    def _update_goal_action(self):
        """Scoring state machine (`line_up` -> `face_goal` -> `charge` -> success).

        - `line_up`: move onto the goal centerline that passes through scoring point.
        - `face_goal`: rotate in place to match scoring heading.
        - `charge`: drive to scoring point while maintaining heading.
        - success: requires arriving at target pose (distance + heading + lateral alignment).
        """
        active_target = self._goal_score_target if self._goal_action_mode == "score" else self._goal_block_target
        target_pos, target_heading, goal, entry_side, opponent_robot = active_target
        dx = target_pos[0] - self.body.position.x
        dy = target_pos[1] - self.body.position.y
        center_distance = math.hypot(dx, dy)
        heading_error = abs(self._normalize_angle(target_heading - self.body.angle))
        goal_dir_x = math.cos(target_heading)
        goal_dir_y = math.sin(target_heading)
        line_normal_x = -goal_dir_y
        line_normal_y = goal_dir_x

        rel_x = self.body.position.x - target_pos[0]
        rel_y = self.body.position.y - target_pos[1]
        lateral_error = rel_x * line_normal_x + rel_y * line_normal_y
        line_up_target = (
            self.body.position.x - lateral_error * line_normal_x,
            self.body.position.y - lateral_error * line_normal_y,
        )
        line_up_tolerance = ROBOT_ARRIVAL_EPSILON

        if (
            center_distance <= ROBOT_ARRIVAL_EPSILON
            and abs(lateral_error) <= line_up_tolerance
            and heading_error <= ROBOT_HEADING_EPSILON
        ):
            self.body.velocity = (0, 0)
            self.body.angular_velocity = 0
            self._goal_action_phase = None
            self._goal_score_target = None
            self._goal_block_target = None

            if self._goal_action_mode == "score":
                # at position to score, try to push ball into goal.
                opponent_pos = (opponent_robot.body.position.x, opponent_robot.body.position.y)
                selected_index = random.randrange(len(self.inventory))
                selected_ball = self.inventory[selected_index]
                scored = goal.try_score(
                    ball=selected_ball,
                    entry_side=entry_side,
                    blocker_pos=opponent_pos,
                )
                if scored:
                    self.inventory.pop(selected_index)
                    return True
                return False

            self._goal_action_mode = None
            return True

        if self._goal_action_phase == "line_up":
            self._apply_motion(line_up_target, self.body.angle)
            if abs(lateral_error) <= line_up_tolerance:
                self._goal_action_phase = "face_goal"
            return False

        if self._goal_action_phase == "face_goal":
            if abs(lateral_error) > line_up_tolerance:
                self._goal_action_phase = "line_up"
                return False
            self._apply_motion((self.body.position.x, self.body.position.y), target_heading)
            if heading_error <= ROBOT_HEADING_EPSILON:
                self._goal_action_phase = "charge"
            return False

        if abs(lateral_error) > line_up_tolerance:
            self._goal_action_phase = "line_up"
            return False

        if heading_error > ROBOT_HEADING_EPSILON:
            self._goal_action_phase = "face_goal"
            return False

        self._apply_motion(target_pos, target_heading)
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

        if (self._goal_score_target is not None or self._goal_block_target is not None) and self._goal_action_phase is not None:
            self._update_goal_action()
            return

        if self.move_target_pos is None or self.move_target_angle is None:
            self.body.velocity = (0, 0)
            self.body.angular_velocity = 0
            return

        distance, angle_error = self._apply_motion(self.move_target_pos, self.move_target_angle)
        if distance <= ROBOT_ARRIVAL_EPSILON and angle_error <= ROBOT_HEADING_EPSILON:
            self.stop()
            
        for ball in self.inventory:
            # if ball is in inventory, update its position to match robot's position. Observation convenience.
            ball.body.position = self.body.position


class Ball:
    STATE_GROUND = "ground"
    STATE_LONG_GOAL_1 = "long_goal_1"
    STATE_LONG_GOAL_2 = "long_goal_2"
    STATE_CENTER_LOWER = "center_lower"
    STATE_CENTER_UPPER = "center_upper"
    STATE_ROBOT_RED = "robot_red"
    STATE_ROBOT_BLUE = "robot_blue"

    COLORS = {
        "red": (220, 40, 40),
        "blue": (40, 80, 220),
    }

    def __init__(self, space, position, color="red"):
        color_key = color.lower()
        assert color_key in self.COLORS, "Color must be 'red' or 'blue'."
        moment = pymunk.moment_for_circle(BALL_MASS_KG, 0, BALL_RADIUS)
        self.body = pymunk.Body(mass=BALL_MASS_KG, moment=moment, body_type=pymunk.Body.DYNAMIC)
        self.body.position = position
        self.body.velocity_func = self._apply_rolling_resistance
        self.shape = pymunk.Circle(self.body, BALL_RADIUS)
        self.shape.friction = 0.7
        self.shape.elasticity = 0.1
        ball_category = CATEGORY_BALL_RED if color_key == "red" else CATEGORY_BALL_BLUE
        self.shape.filter = pymunk.ShapeFilter(categories=ball_category, mask=pymunk.ShapeFilter.ALL_MASKS())
        self.color = self.COLORS[color_key]
        self.color_key = color_key
        self.state = self.STATE_GROUND 
        self.relative_to = {} # dict to keep relative metric to robot. used for observations.

        space.add(self.body, self.shape)

    def update_relative_to_robot(self, robot, robot_key="robot_red"):
        """Update relative metric to robot after every sim step. Metric includes:
        - dx, dy: ball position relative to robot position in world frame.
        - distance: center distance between ball and robot.
        - delta_theta: angle difference between robot heading and ball relative position."""
        dx = self.body.position.x - robot.body.position.x
        dy = self.body.position.y - robot.body.position.y
        relative_distance = math.hypot(dx, dy)
        theta = math.atan2(dy, dx)
        robot_theta = robot.body.angle
        delta_theta = (theta - robot_theta + math.pi) % (2 * math.pi) - math.pi

        self.relative_to[robot_key] = {
            "dx": dx,
            "dy": dy,
            "distance": relative_distance,
            "delta_theta": delta_theta,
        }
        return self.relative_to[robot_key]

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


class Goal:
    def __init__(self, space, position, length, width, angle_deg, color=(255, 220, 0), capacity=0):
        assert capacity >= 0, "Goal capacity cannot be negative."
        self.body = pymunk.Body(body_type=pymunk.Body.STATIC)
        self.body.position = position
        self.body.angle = math.radians(angle_deg)
        self.shape = pymunk.Poly.create_box(self.body, size=(length, width))
        self.shape.friction = 1.0
        self.shape.elasticity = 0.0
        self.color = color
        self.length = length
        self.width = width
        self.capacity = capacity
        self.scored_balls = [None] * capacity if capacity > 0 else []
        space.add(self.body, self.shape)

    def is_full(self):
        """Check full goal.
        """
        return sum(ball is not None for ball in self.scored_balls) == self.capacity

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

    def _entry_scan_indices(self, entry_side: int):
        if entry_side == 0:
            return range(self.capacity)
        return range(self.capacity - 1, -1, -1)

    def _is_output_blocked(self, entry_side: int, blocker_pos):
        output_side = 1 if entry_side == 0 else 0
        output_pos = self.scoring_position[output_side]
        dx = blocker_pos[0] - output_pos[0]
        dy = blocker_pos[1] - output_pos[1]
        return math.hypot(dx, dy) <= GOAL_BLOCK_EPSILON_CM

    def can_accept(self, entry_side: int, blocker_pos=None):
        if not self.is_full():
            return True
        return not self._is_output_blocked(entry_side, blocker_pos)

    def try_score(self, ball, entry_side: int, blocker_pos=None):
        if self.is_full():
            if entry_side == 0:
                eject_indices = range(self.capacity - 1, -1, -1)
            else:
                eject_indices = range(self.capacity)

            ejected_slot = None
            for idx in eject_indices:
                if self.scored_balls[idx] is not None:
                    ejected_slot = idx
                    break

            if ejected_slot is None:
                return False

            ejected_ball = self.scored_balls[ejected_slot]
            self.scored_balls[ejected_slot] = None
            if hasattr(ejected_ball, "state"):
                ejected_ball.state = Ball.STATE_GROUND

        insert_slot = None
        for idx in self._entry_scan_indices(entry_side):
            if self.scored_balls[idx] is None:
                insert_slot = idx
                break

        if insert_slot is None:
            return False

        self.scored_balls[insert_slot] = ball

        if hasattr(ball, "state"):
            goal_to_state = {
                "long_1": Ball.STATE_LONG_GOAL_1,
                "long_2": Ball.STATE_LONG_GOAL_2,
                "center_lower": Ball.STATE_CENTER_LOWER,
                "center_upper": Ball.STATE_CENTER_UPPER,
            }
            ball.state = goal_to_state.get(getattr(self, "goal_key", None), Ball.STATE_GROUND)

        for slot_idx, slot_ball in enumerate(self.scored_balls):
            if slot_ball is None:
                continue
            slot_pos = self._slot_world_position(slot_idx)
            slot_ball.body.position = slot_pos
            slot_ball.body.velocity = (0, 0)
            slot_ball.body.angular_velocity = 0
        return True


class CenterGoalLower(Goal):
    def __init__(self, space):
        super().__init__(
            space=space,
            position=(FIELD_WIDTH / 2, FIELD_HEIGHT / 2),
            length=CENTER_GOAL_LOWER_LENGTH,
            width=CENTER_GOAL_LOWER_WIDTH,
            angle_deg=45,
            color=(255, 220, 0),
            capacity=7,
        )
        half_length = CENTER_GOAL_LOWER_LENGTH / 2
        half_robot = ROBOT_SIZE / 2
        clearance = half_length + half_robot + 2
        angle = math.radians(45)
        cos = math.cos(angle)
        self.scoring_position = [
            (
                FIELD_WIDTH / 2 - cos * clearance,
                FIELD_HEIGHT / 2 - cos * clearance,
            ),
            (
                FIELD_WIDTH / 2 + cos * clearance,
                FIELD_HEIGHT / 2 + cos * clearance,
            ),
        ]
        self.goal_key = "center_lower"
        self.score_side = [0,1]


class CenterGoalUpper(Goal):
    def __init__(self, space):
        super().__init__(
            space=space,
            position=(FIELD_WIDTH / 2, FIELD_HEIGHT / 2),
            length=CENTER_GOAL_UPPER_LENGTH,
            width=CENTER_GOAL_UPPER_WIDTH,
            angle_deg=-45,
            color=(255, 220, 0),
            capacity=7,
        )
        self.shape.filter = pymunk.ShapeFilter(categories=CATEGORY_GOAL_UPPER, mask=CATEGORY_ROBOT)
        half_length = CENTER_GOAL_UPPER_LENGTH / 2
        half_robot = ROBOT_SIZE / 2
        clearance = half_length + half_robot + 2
        angle = math.radians(45)
        cos = math.cos(angle)
        self.scoring_position = [
            (
                FIELD_WIDTH / 2 - cos * clearance,
                FIELD_HEIGHT / 2 + cos * clearance,
            ),
            (
                FIELD_WIDTH / 2 + cos * clearance,
                FIELD_HEIGHT / 2 - cos * clearance,
            ),
        ]
        self.goal_key = "center_upper"
        self.score_side = [0, 1]


class LongGoal(Goal):
    def __init__(self, space, position):
        super().__init__(
            space=space,
            position=position,
            length=LONG_GOAL_LENGTH,
            width=LONG_GOAL_WIDTH,
            angle_deg=0,
            color=(255, 220, 0),
            capacity=15,
        )
        self.shape.filter = pymunk.ShapeFilter(categories=CATEGORY_GOAL_UPPER, mask=CATEGORY_ROBOT)
        half_length = LONG_GOAL_LENGTH / 2 + 2
        half_robot = ROBOT_SIZE / 2
        self.scoring_position = [(position[0] - half_length - half_robot, position[1]), (position[0] + half_length + half_robot, position[1])]
        self.score_side = [0,1]


class LongGoalOne(LongGoal):
    def __init__(self, space):
        super().__init__(space=space, position=LONG_GOAL_1_POSITION)
        self.goal_key = "long_1"


class LongGoalTwo(LongGoal):
    def __init__(self, space):
        super().__init__(space=space, position=LONG_GOAL_2_POSITION)
        self.goal_key = "long_2"


class LongGoalLeg(Goal):
    def __init__(self, space, position):
        super().__init__(
            space=space,
            position=position,
            length=LONG_GOAL_LEG_SIZE,
            width=LONG_GOAL_LEG_SIZE,
            angle_deg=0,
            color=(255, 220, 0),
        )

class LongGoalOneLegOne(LongGoalLeg):
    def __init__(self, space):
        super().__init__(space=space, position=(LONG_GOAL_LEG_X1, LONG_GOAL_1_POSITION[1]))


class LongGoalOneLegTwo(LongGoalLeg):
    def __init__(self, space):
        super().__init__(space=space, position=(LONG_GOAL_LEG_X2, LONG_GOAL_1_POSITION[1]))


class LongGoalTwoLegOne(LongGoalLeg):
    def __init__(self, space):
        super().__init__(space=space, position=(LONG_GOAL_LEG_X1, LONG_GOAL_2_POSITION[1]))


class LongGoalTwoLegTwo(LongGoalLeg):
    def __init__(self, space):
        super().__init__(space=space, position=(LONG_GOAL_LEG_X2, LONG_GOAL_2_POSITION[1]))


def step_space(space, dt):
    substeps = 4
    sub_dt = dt / substeps
    for _ in range(substeps):
        space.step(sub_dt)
        
