import pygame
import pymunk
import math
from config import VexConfig
from env.type import Field
INCH_TO_CM = 2.54
GRID_SPACING_INCH = 3.23
SHOW_REGION = True
GOAL_FILL_ALPHA = 95
GOAL_BORDER_ALPHA = 180
SCORED_BALL_ALPHA = 150
TEAM_COLOURS = {
    "red": (220, 40, 40),
    "blue": (40, 80, 220),
}
GOAL_NEUTRAL_COLOUR = (255, 220, 0)


class EnvRenderer:
    def __init__(self, config: VexConfig):
        pygame.init()
        main_config = config
        engine_config = main_config.engine_config
        self.field_view_width = main_config.window_width
        self.window_height = main_config.window_height
        self.inference_hz = main_config.inference_hz
        self.max_duration_s = main_config.max_duration_s
        self.info_panel_width = max(220, int(self.field_view_width * 0.22))
        self.window_width = self.field_view_width + self.info_panel_width
        self.scale_x = self.field_view_width / engine_config['field']['width']
        self.scale_y = self.window_height / engine_config['field']['height']
        self.pickup_dist_threshold = main_config.ball_pickup_hitbox["dist_threshold"]
        self.pickup_angle_threshold_rad = main_config.ball_pickup_hitbox["angle_threshold"]
        self.goal_dist_threshold = main_config.goal_action_hitbox["dist_threshold"]
        self.goal_angle_threshold_rad = main_config.goal_action_hitbox["angle_threshold"]
        self.loader_dist_threshold = main_config.loader_pickup_hitbox["dist_threshold"]
        self.screen = pygame.display.set_mode((self.window_width, self.window_height))
        self.font = pygame.font.SysFont("consolas", 20)
        self._static_field_surface = None
        self._scored_balls_surface = None
        self._scored_balls_signature = None
        pygame.display.set_caption("Vex RL Env")
        
        self.field_width = engine_config['field']['width']
        self.field_height = engine_config['field']['height']
        self.ball_radius = engine_config['ball']['radius']
        self.robot_size = engine_config['robot']['size']

    def _world_to_screen(self, point):
        return (
            int(point[0] * self.scale_x),
            int(self.window_height - point[1] * self.scale_y),
        )

    def _draw_grid(self):
        step = GRID_SPACING_INCH * INCH_TO_CM

        x = 0.0
        while x <= self.field_width:
            x_screen = int(x * self.scale_x)
            pygame.draw.line(self.screen, (195, 195, 195), (x_screen, 0), (x_screen, self.window_height), 1)
            x += step

        y = 0.0
        while y <= self.field_height:
            y_screen = int(self.window_height - y * self.scale_y)
            pygame.draw.line(self.screen, (195, 195, 195), (0, y_screen), (self.field_view_width, y_screen), 1)
            y += step

    def _draw_info_panel(self, field_dict):
        panel_x = self.field_view_width
        panel_rect = pygame.Rect(panel_x, 0, self.info_panel_width, self.window_height)
        pygame.draw.rect(self.screen, (235, 235, 235), panel_rect)
        pygame.draw.line(self.screen, (120, 120, 120), (panel_x, 0), (panel_x, self.window_height), 2)

        elapsed_time_s = field_dict.get("elapsed_time_s")
        if elapsed_time_s is None:
            actions_counter = int(field_dict.get("actions_counter", 0))
            elapsed_time_s = actions_counter / self.inference_hz if self.inference_hz > 0 else 0.0
        else:
            elapsed_time_s = float(elapsed_time_s)

        max_duration_s = field_dict.get("max_duration_s")
        if max_duration_s is None:
            max_duration_s = self.max_duration_s
        else:
            max_duration_s = float(max_duration_s)

        if max_duration_s > 0:
            elapsed_time_s = min(elapsed_time_s, max_duration_s)

        elapsed_int = max(0, int(elapsed_time_s))
        max_int = max(0, int(max_duration_s))
        elapsed_label = f"{elapsed_int // 60:02d}:{elapsed_int % 60:02d}"
        max_label = f"{max_int // 60:02d}:{max_int % 60:02d}"

        red_inv = field_dict["red_robot"].inventory
        blue_inv = field_dict["blue_robot"].inventory
        red_red = sum(1 for ball in red_inv if getattr(ball, "colour", None) == "red")
        blue_blue = sum(1 for ball in blue_inv if getattr(ball, "colour", None) == "blue")
        red_score = field_dict.get("red_score", 0)
        blue_score = field_dict.get("blue_score", 0)
        loader_lines = []
        for loader_key in ("LD1", "LD2", "LD3", "LD4"):
            loader = field_dict.get(loader_key)
            if loader is None:
                continue
            slots = []
            red_count = 0
            blue_count = 0
            for slot_ball in getattr(loader, "scored_balls", []):
                if slot_ball is None:
                    slots.append("0")
                    continue
                ball_colour = getattr(slot_ball, "colour", None)
                if ball_colour == "red":
                    slots.append("R")
                    red_count += 1
                elif ball_colour == "blue":
                    slots.append("B")
                    blue_count += 1
                else:
                    slots.append("?")
            slot_signature = "".join(slots)
            loader_lines.append((f"{loader_key}: {slot_signature}", (80, 80, 80)))
            loader_lines.append((f"  r/b: {red_count}/{blue_count}", (120, 120, 120)))

        manager_lines = []
        manager_entries = {}
        for loader in field_dict.get("loaders", []):
            manager = getattr(loader, "manager", None)
            if manager is None:
                continue
            manager_id = id(manager)
            goal_key = str(getattr(loader, "goal_key", ""))
            loader_suffix = goal_key.split("_")[-1] if "_" in goal_key else "?"
            if manager_id not in manager_entries:
                manager_entries[manager_id] = {
                    "manager": manager,
                    "loader_suffixes": [],
                }
            manager_entries[manager_id]["loader_suffixes"].append(loader_suffix)

        for entry in manager_entries.values():
            manager = entry["manager"]
            raw_suffixes = entry["loader_suffixes"]
            if all(suffix.isdigit() for suffix in raw_suffixes):
                sorted_suffixes = [str(value) for value in sorted({int(suffix) for suffix in raw_suffixes})]
            else:
                sorted_suffixes = sorted(set(raw_suffixes))

            manager_colour = getattr(manager, "colour", None)
            left_to_load = getattr(manager, "left_to_load", len(getattr(manager, "inventory", [])))
            manager_name = f"LM{''.join(sorted_suffixes)}" if len(sorted_suffixes) > 0 else "LM?"
            if manager_colour in ("red", "blue"):
                manager_name += f" ({manager_colour})"
            line_colour = TEAM_COLOURS.get(manager_colour, (80, 80, 80))
            manager_lines.append((f"{manager_name}: {left_to_load} left", line_colour))

        lines = [
            (f"Time: {elapsed_label} / {max_label}", (30, 30, 30)),
            ("", (30, 30, 30)),
            ("Inventory", (30, 30, 30)),
            ("", (30, 30, 30)),
            (f"Red: {len(red_inv)}", (220, 40, 40)),
            (f"  red balls:  {red_red}", (80, 80, 80)),
            ("", (30, 30, 30)),
            (f"Blue: {len(blue_inv)}", (40, 80, 220)),
            (f"  blue balls: {blue_blue}", (80, 80, 80)),
            (f"Score: R {red_score} - B {blue_score}", (30, 30, 30)),
        ]

        if len(loader_lines) > 0:
            lines.append(("", (30, 30, 30)))
            lines.append(("Loaders", (30, 30, 30)))
            lines.extend(loader_lines)

        if len(manager_lines) > 0:
            lines.append(("", (30, 30, 30)))
            lines.append(("Loader Managers", (30, 30, 30)))
            lines.extend(manager_lines)

        y = 20
        for text, color in lines:
            if text == "":
                y += 12
                continue
            surface = self.font.render(text, True, color)
            self.screen.blit(surface, (panel_x + 14, y))
            y += 26

    def _draw_wall(self, wall):
        for segment in wall.shapes:
            a = segment.a.rotated(segment.body.angle) + segment.body.position
            b = segment.b.rotated(segment.body.angle) + segment.body.position
            a_screen = self._world_to_screen((a.x, a.y))
            b_screen = self._world_to_screen((b.x, b.y))
            width = max(1, int(segment.radius * 2 * self.scale_x))
            pygame.draw.line(self.screen, (60, 60, 60), a_screen, b_screen, width)

    def _draw_goal(self, goal):
        points_world = [p.rotated(goal.body.angle) + goal.body.position for p in goal.shape.get_vertices()]
        points_screen = [self._world_to_screen((p.x, p.y)) for p in points_world]
        overlay = pygame.Surface((self.window_width, self.window_height), pygame.SRCALPHA)
        pygame.draw.polygon(overlay, (*GOAL_NEUTRAL_COLOUR, GOAL_FILL_ALPHA), points_screen)
        pygame.draw.polygon(overlay, (*GOAL_NEUTRAL_COLOUR, GOAL_BORDER_ALPHA), points_screen, 2)
        self.screen.blit(overlay, (0, 0))

    def _build_static_field_surface(self, field_dict):
        static_surface = pygame.Surface((self.window_width, self.window_height))
        static_surface.fill((217, 217, 217))

        original_screen = self.screen
        try:
            self.screen = static_surface
            self._draw_grid()
            self._draw_wall(field_dict["wall"])
            self._draw_goal(field_dict["center_lower"])
            self._draw_goal(field_dict["center_upper"])
            self._draw_goal(field_dict["long_1"])
            self._draw_goal(field_dict["long_2"])
            for loader in field_dict.get("loaders", []):
                self._draw_goal(loader)
            if SHOW_REGION:
                self._draw_score_regions(field_dict)
                self._draw_loader_regions(field_dict)
        finally:
            self.screen = original_screen

        self._static_field_surface = static_surface

    def _get_scored_balls_signature(self, field_dict):
        goals = [field_dict["center_lower"], field_dict["center_upper"], field_dict["long_1"], field_dict["long_2"]]
        signature = []
        for goal in goals:
            goal_slots = []
            for scored_ball in goal.scored_balls:
                if scored_ball is None:
                    goal_slots.append("0")
                else:
                    goal_slots.append(getattr(scored_ball, "colour", "?"))
            signature.append(tuple(goal_slots))
        return tuple(signature)

    def _refresh_scored_balls_surface_if_needed(self, field_dict):
        scored_signature = self._get_scored_balls_signature(field_dict)
        if self._scored_balls_surface is not None and scored_signature == self._scored_balls_signature:
            return

        scored_surface = pygame.Surface((self.window_width, self.window_height), pygame.SRCALPHA)
        self._draw_scored_balls(field_dict["center_lower"], scored_surface)
        self._draw_scored_balls(field_dict["center_upper"], scored_surface)
        self._draw_scored_balls(field_dict["long_1"], scored_surface)
        self._draw_scored_balls(field_dict["long_2"], scored_surface)
        self._scored_balls_surface = scored_surface
        self._scored_balls_signature = scored_signature

    def _draw_scored_balls(self, goal, target_surface):
        if not hasattr(goal, "scored_balls"):
            return

        radius_px = max(2, int(self.ball_radius * 0.7 * self.scale_x))

        for scored_ball in goal.scored_balls:
            if scored_ball is None:
                continue
            center = self._world_to_screen((scored_ball.body.position.x, scored_ball.body.position.y))
            if hasattr(scored_ball, "colour"):
                render_color = TEAM_COLOURS.get(scored_ball.colour, (30, 30, 30))
            else:
                render_color = TEAM_COLOURS.get(scored_ball, (30, 30, 30))
            pygame.draw.circle(target_surface, (*render_color, SCORED_BALL_ALPHA), center, radius_px)
            pygame.draw.circle(target_surface, (0, 0, 0, 160), center, radius_px, 1)

    def _draw_ball(self, ball):
        center = self._world_to_screen((ball.body.position.x, ball.body.position.y))
        render_color = TEAM_COLOURS.get(getattr(ball, "colour", None), (30, 30, 30))
        pygame.draw.circle(self.screen, render_color, center, max(1, int(self.ball_radius * self.scale_x)))

    def _draw_robot(self, robot, body_color):
        body_points_world = [p.rotated(robot.body.angle) + robot.body.position for p in robot.shape.get_vertices()]
        body_points_screen = [self._world_to_screen((p.x, p.y)) for p in body_points_world]
        pygame.draw.polygon(self.screen, body_color, body_points_screen)

        side = self.robot_size
        cell_half = side / 6
        front_center = pymunk.Vec2d(side / 3, 0)
        front_tile_local = [
            front_center + (-cell_half, -cell_half),
            front_center + (cell_half, -cell_half),
            front_center + (cell_half, cell_half),
            front_center + (-cell_half, cell_half),
        ]
        front_tile_world = [p.rotated(robot.body.angle) + robot.body.position for p in front_tile_local]
        front_tile_screen = [self._world_to_screen((p.x, p.y)) for p in front_tile_world]
        pygame.draw.polygon(self.screen, (255, 200, 0), front_tile_screen)

    def _draw_pickup_region(self, robot, color):
        center = robot.body.position
        heading = robot.body.angle
        start_angle = heading - self.pickup_angle_threshold_rad
        end_angle = heading + self.pickup_angle_threshold_rad

        arc_points = [self._world_to_screen((center.x, center.y))]
        arc_samples = 32
        for i in range(arc_samples + 1):
            t = i / arc_samples
            a = start_angle + (end_angle - start_angle) * t
            px = center.x + math.cos(a) * self.pickup_dist_threshold
            py = center.y + math.sin(a) * self.pickup_dist_threshold
            arc_points.append(self._world_to_screen((px, py)))

        overlay = pygame.Surface((self.window_width, self.window_height), pygame.SRCALPHA)
        pygame.draw.polygon(overlay, (*color, 60), arc_points)
        pygame.draw.polygon(overlay, (*color, 150), arc_points, 2)
        self.screen.blit(overlay, (0, 0))

    def _draw_score_regions(self, field_dict):
        goals = [field_dict["center_lower"], field_dict["center_upper"], field_dict["long_1"], field_dict["long_2"]]
        radius_px = max(1, int(self.goal_dist_threshold * self.scale_x))
        overlay = pygame.Surface((self.window_width, self.window_height), pygame.SRCALPHA)

        for goal in goals:
            for scoring_position in goal.scoring_position:
                center = self._world_to_screen(scoring_position)
                pygame.draw.circle(overlay, (30, 170, 60, 30), center, radius_px)
                pygame.draw.circle(overlay, (30, 170, 60, 180), center, radius_px, 2)

        self.screen.blit(overlay, (0, 0))

    def _draw_loader_regions(self, field_dict):
        radius_px = max(1, int(self.loader_dist_threshold * self.scale_x))
        overlay = pygame.Surface((self.window_width, self.window_height), pygame.SRCALPHA)

        for loader in field_dict.get("loaders", []):
            center = self._world_to_screen(loader.loading_position)
            pygame.draw.circle(overlay, (180, 120, 30, 30), center, radius_px)
            pygame.draw.circle(overlay, (180, 120, 30, 180), center, radius_px, 2)

        self.screen.blit(overlay, (0, 0))

    def render(self, field: Field):
        field_dict = field.to_field_dict()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return None

        if self._static_field_surface is None:
            self._build_static_field_surface(field_dict)

        self.screen.blit(self._static_field_surface, (0, 0))
        self._refresh_scored_balls_surface_if_needed(field_dict)
        self.screen.blit(self._scored_balls_surface, (0, 0))

        if SHOW_REGION:
            self._draw_pickup_region(field_dict["red_robot"], (220, 40, 40))
            self._draw_pickup_region(field_dict["blue_robot"], (40, 80, 220))

        for ball in field_dict["balls"]:
            if ball.state != "ground":
                continue
            self._draw_ball(ball)

        self._draw_robot(field_dict["red_robot"], (220, 40, 40))
        self._draw_robot(field_dict["blue_robot"], (40, 80, 220))
        self._draw_info_panel(field_dict)

        pygame.display.update()
        return None

    def close(self):
        pygame.quit()
