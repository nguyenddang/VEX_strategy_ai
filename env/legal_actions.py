from typing import Any, Callable, Dict, List
import math
from env.engine_core.utils import normalize_angle
from env.engine_core.field_component import Goal, Loader, Ball
from env.engine_core.robot import Robot
from env.type import Field

class LegalActionResolver:
    def __init__(
        self,
        goal_action_hitbox: Dict[str, float],
        loader_pickup_hitbox: Dict[str, float],
        ball_pickup_hitbox: Dict[str, float],
    ):
        self.goal_action_hitbox = goal_action_hitbox
        self.loader_pickup_hitbox = loader_pickup_hitbox
        self.ball_pickup_hitbox = ball_pickup_hitbox

    def find_nearest_building_target(self, robot: Robot, opponent_robot: Robot, goals: List[Goal]):
        """Find goal to score or block based on proximity and angle to scoring positions.

        Assumes scoring/blocking regions do not overlap; therefore at most one scoring
        and one blocking target exist, and they reference the same goal.
        """
        goal_dist_threshold = self.goal_action_hitbox["dist_threshold"]
        goal_angle_threshold = self.goal_action_hitbox["angle_threshold"]
        blocker_pose = opponent_robot.cache_pose
        curr_pose = robot.cache_pose
        score_target, block_target = None, None
        for goal in goals:
            goal._update_relative_to_robot(robot)
        for goal in goals:
            relative = goal.relative_stats[robot.key]
            for side_index, rlt in enumerate(relative):
                target_heading = math.atan2(
                    goal.cache_pose['position'][1] - goal.scoring_position[side_index][1],
                    goal.cache_pose['position'][0] - goal.scoring_position[side_index][0],
                )
                heading_error = normalize_angle(target_heading - curr_pose['angle'])
                within_goal_hitbox = (
                    rlt['distance'] <= goal_dist_threshold and abs(heading_error) <= goal_angle_threshold
                )
                if not within_goal_hitbox:
                    continue

                block_target = (goal.scoring_position[side_index], target_heading, goal, side_index)
                if goal.can_accept(entry_side=side_index, blocker_pos=(blocker_pose['position'][0], blocker_pose['position'][1])) and len(robot.inventory) > 0:
                    score_target = (goal.scoring_position[side_index], target_heading, goal, side_index)
                return score_target, block_target
        return score_target, block_target

    def find_nearest_loader_pickup_target(self, robot: Robot, loaders: List[Loader]):
        """Find loader to pickup from based on proximity and angle to loading position.
        
        Assumes loader pickup regions do not overlap; therefore at most one loader pickup
        target exists.
        """
        loader_dist_threshold = self.loader_pickup_hitbox["dist_threshold"]
        loader_angle_threshold = self.loader_pickup_hitbox["angle_threshold"]
        for loader in loaders:
            loader._update_relative_to_robot(robot)
        for loader in loaders:
            if loader.scored_balls[0] is None:
                continue
            loading_position = loader.loading_position
            relative = loader.relative_stats[robot.key]
            target_heading = math.atan2(
                loader.cache_pose['position'][1] - loading_position[1],
                loader.cache_pose['position'][0] - loading_position[0],
            )
            within_loader_hitbox = (
                relative["distance"] <= loader_dist_threshold and abs(relative["delta_theta"]) <= loader_angle_threshold
            )
            if not within_loader_hitbox:
                continue
            return (loading_position, target_heading, loader)
        return None
    
    def find_nearest_ball_pickup_target(self, robot: Robot, balls: List[Ball]):
        """Find ball to pickup based on proximity and angle to ball position for single robot, and assign pickup candidates.
        
        Update relative position of each ball to robot. To be used to build observations. 
        """
        
        min_dist = float("inf")
        dist_threshold = self.ball_pickup_hitbox["dist_threshold"]
        angle_threshold = self.ball_pickup_hitbox["angle_threshold"]
        pickup_ball = None
        for ball in balls:
            relative = ball.update_relative_to_robot(robot)
            if ball.state != "ground":
                continue
            # determine pickup-able ball within hitbox for each robot. 
            ball_colour = ball.colour
            can_pickup = (
                relative["distance"] <= dist_threshold
                and abs(relative["delta_theta"]) <= angle_threshold
                and len(robot.inventory) < robot.capacity
            )
            if can_pickup:
                if relative["distance"] < min_dist and ball_colour == robot.key.split('_')[-1]:
                    pickup_ball = ball
                    min_dist = relative["distance"] 
        return pickup_ball
    
    def get_legal_actions(self, field: Field) -> Dict[str, Any]:
        legal_actions: Dict[str, Any] = {}

        for robot in [field.robot_red, field.robot_blue]:
            opponent_robot = field.robot_blue if robot.key == "robot_red" else field.robot_red
            score_target, block_target = self.find_nearest_building_target(robot=robot,opponent_robot=opponent_robot,goals=field.goals,)
            loader_target = self.find_nearest_loader_pickup_target(robot, field.loaders)
            pickup_ball = self.find_nearest_ball_pickup_target(robot, field.balls)
            can_score = score_target is not None
            can_block = block_target is not None
            can_pickup_loader = loader_target is not None
            can_pickup_ground = pickup_ball is not None
            discrete_mask = [True, True, can_pickup_loader, can_pickup_ground, can_score, can_block]
            robot._building_score_target = score_target
            robot._building_block_target = block_target
            robot._building_loader_target = loader_target
            robot._pickup_ball = pickup_ball
            legal_actions[robot.key] = discrete_mask
        return legal_actions
