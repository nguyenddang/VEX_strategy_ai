from typing import Any, Dict, List

from pydantic import BaseModel, ConfigDict, model_validator

from env.engine_core.field_component import Ball, Goal, Leg, Loader, Wall
from env.engine_core.robot import Robot


_REQUIRED_GOAL_KEYS = {"center_lower", "center_upper", "long_1", "long_2"}
_REQUIRED_LOADER_KEYS = {"loader_1", "loader_2", "loader_3", "loader_4"}


class Field(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    wall: Wall
    legs: List[Leg]
    goals: List[Goal]
    loaders: List[Loader]
    robot_red: Robot
    robot_blue: Robot
    balls: List[Ball]
    actions_counter: int = 0
    red_score: int = 0
    blue_score: int = 0

    @model_validator(mode="after")
    def _validate_required_keys(self):
        goal_keys = {goal.key for goal in self.goals}
        missing_goals = _REQUIRED_GOAL_KEYS - goal_keys
        if missing_goals:
            raise ValueError(f"Missing required goals in field_dict: {sorted(missing_goals)}")

        loader_keys = {loader.key for loader in self.loaders}
        missing_loaders = _REQUIRED_LOADER_KEYS - loader_keys
        if missing_loaders:
            raise ValueError(f"Missing required loaders in field_dict: {sorted(missing_loaders)}")

        return self

    def to_field_dict(self) -> Dict[str, Any]:
        goal_by_key = {goal.key: goal for goal in self.goals}
        loader_by_key = {loader.key: loader for loader in self.loaders}

        return {
            "wall": self.wall,
            "legs": self.legs,
            "goals": self.goals,
            "loaders": self.loaders,
            "robot_red": self.robot_red,
            "robot_blue": self.robot_blue,
            "red_robot": self.robot_red,
            "blue_robot": self.robot_blue,
            "balls": self.balls,
            "CGL": goal_by_key["center_lower"],
            "CGU": goal_by_key["center_upper"],
            "LG1": goal_by_key["long_1"],
            "LG2": goal_by_key["long_2"],
            "LD1": loader_by_key["loader_1"],
            "LD2": loader_by_key["loader_2"],
            "LD3": loader_by_key["loader_3"],
            "LD4": loader_by_key["loader_4"],
            'actions_counter': self.actions_counter,
            'red_score': self.red_score,
            'blue_score': self.blue_score,
        }
