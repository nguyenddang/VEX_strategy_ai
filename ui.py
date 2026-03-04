import pygame
import pymunk
import math

from env.engine import FIELD_SIZE, ROBOT_SIZE, WALL_THICKNESS, BALL_RADIUS, INITIAL_BALL_POSITIONS_CM, INITIAL_RED_ROBOT_POSITION_CM, INITIAL_BLUE_ROBOT_POSITION_CM, create_space, Wall, Robot, Ball, CenterGoalLower, CenterGoalUpper, LongGoalOne, LongGoalTwo, LongGoalOneLegOne, LongGoalOneLegTwo, LongGoalTwoLegOne, LongGoalTwoLegTwo, step_space


WINDOW_WIDTH, WINDOW_HEIGHT = 1200, 1200
SCALE_X = WINDOW_WIDTH / FIELD_SIZE
SCALE_Y = WINDOW_HEIGHT / FIELD_SIZE
INCH_TO_CM = 2.54
GRID_SPACING_INCH = 3.23
REQUERY_INTERVAL_S = 0.2


def world_to_screen(point):
    return (int(point[0] * SCALE_X), int(point[1] * SCALE_Y))


def screen_to_world(point):
    return (point[0] / SCALE_X, point[1] / SCALE_Y)


def draw_robot(screen, robot, body_color):
    body_points_world = [p.rotated(robot.body.angle) + robot.body.position for p in robot.shape.get_vertices()]
    body_points_screen = [world_to_screen((p.x, p.y)) for p in body_points_world]
    pygame.draw.polygon(screen, body_color, body_points_screen)

    side = ROBOT_SIZE
    cell_half = side / 6
    front_center = pymunk.Vec2d(side / 3, 0)
    front_tile_local = [
        front_center + (-cell_half, -cell_half),
        front_center + (cell_half, -cell_half),
        front_center + (cell_half, cell_half),
        front_center + (-cell_half, cell_half),
    ]
    front_tile_world = [p.rotated(robot.body.angle) + robot.body.position for p in front_tile_local]
    front_tile_screen = [world_to_screen((p.x, p.y)) for p in front_tile_world]
    pygame.draw.polygon(screen, (255, 200, 0), front_tile_screen)


def draw_wall(screen, wall):
    for segment in wall.shapes:
        a = segment.a.rotated(segment.body.angle) + segment.body.position
        b = segment.b.rotated(segment.body.angle) + segment.body.position
        a_screen = world_to_screen((a.x, a.y))
        b_screen = world_to_screen((b.x, b.y))
        width = max(1, int(segment.radius * 2 * SCALE_X))
        pygame.draw.line(screen, (60, 60, 60), a_screen, b_screen, width)


def draw_ball(screen, ball):
    center = world_to_screen((ball.body.position.x, ball.body.position.y))
    pygame.draw.circle(screen, ball.color, center, max(1, int(ball.radius * SCALE_X)))


def draw_goal(screen, goal):
    points_world = [p.rotated(goal.body.angle) + goal.body.position for p in goal.shape.get_vertices()]
    points_screen = [world_to_screen((p.x, p.y)) for p in points_world]
    pygame.draw.polygon(screen, goal.color, points_screen)


def draw_grid(screen):
    step = GRID_SPACING_INCH * INCH_TO_CM

    x = 0.0
    while x <= FIELD_SIZE:
        x_screen = int(x * SCALE_X)
        pygame.draw.line(screen, (195, 195, 195), (x_screen, 0), (x_screen, WINDOW_HEIGHT), 1)
        x += step

    y = 0.0
    while y <= FIELD_SIZE:
        y_screen = int(y * SCALE_Y)
        pygame.draw.line(screen, (195, 195, 195), (0, y_screen), (WINDOW_WIDTH, y_screen), 1)
        y += step


def draw_target_arrow(screen, origin_world, mouse_world):
    origin_screen = world_to_screen(origin_world)
    mouse_screen = world_to_screen(mouse_world)
    pygame.draw.line(screen, (30, 30, 30), origin_screen, mouse_screen, 2)
    pygame.draw.circle(screen, (30, 30, 30), origin_screen, 4)

    dx = mouse_screen[0] - origin_screen[0]
    dy = mouse_screen[1] - origin_screen[1]
    length = math.hypot(dx, dy)
    if length < 1:
        return

    ux = dx / length
    uy = dy / length
    arrow_len = 12
    wing = 5
    tip = mouse_screen
    left = (int(tip[0] - arrow_len * ux + wing * uy), int(tip[1] - arrow_len * uy - wing * ux))
    right = (int(tip[0] - arrow_len * ux - wing * uy), int(tip[1] - arrow_len * uy + wing * ux))
    pygame.draw.polygon(screen, (30, 30, 30), [tip, left, right])


def render_scene(screen, wall, center_goal_lower, center_goal_upper, long_goal_one, long_goal_two, long_goal_one_leg_one, long_goal_one_leg_two, long_goal_two_leg_one, long_goal_two_leg_two, balls, red_robot, blue_robot, orientation_origin_world=None, orientation_mouse_world=None):
    screen.fill((217, 217, 217))
    draw_grid(screen)
    draw_wall(screen, wall)
    draw_goal(screen, center_goal_lower)
    draw_goal(screen, center_goal_upper)
    draw_goal(screen, long_goal_one)
    draw_goal(screen, long_goal_two)
    draw_goal(screen, long_goal_one_leg_one)
    draw_goal(screen, long_goal_one_leg_two)
    draw_goal(screen, long_goal_two_leg_one)
    draw_goal(screen, long_goal_two_leg_two)
    for ball in balls:
        draw_ball(screen, ball)
    draw_robot(screen, red_robot, (220, 40, 40))
    draw_robot(screen, blue_robot, (40, 80, 220))
    if orientation_origin_world is not None and orientation_mouse_world is not None:
        draw_target_arrow(screen, orientation_origin_world, orientation_mouse_world)
    pygame.display.update()


def run():
    pygame.init()
    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    clock = pygame.time.Clock()

    space = create_space()
    wall = Wall(space)
    center_goal_lower = CenterGoalLower(space)
    center_goal_upper = CenterGoalUpper(space)
    long_goal_one = LongGoalOne(space)
    long_goal_two = LongGoalTwo(space)
    long_goal_one_leg_one = LongGoalOneLegOne(space)
    long_goal_one_leg_two = LongGoalOneLegTwo(space)
    long_goal_two_leg_one = LongGoalTwoLegOne(space)
    long_goal_two_leg_two = LongGoalTwoLegTwo(space)
    red_robot = Robot(space, position=INITIAL_RED_ROBOT_POSITION_CM)
    blue_robot = Robot(space, position=INITIAL_BLUE_ROBOT_POSITION_CM)
    requery_elapsed = 0.0
    pending_target_pose = None
    click_target_pos = None
    selecting_orientation = False
    orientation_mouse_world = None

    balls = []

    for position in INITIAL_BALL_POSITIONS_CM["red"]:
        balls.append(Ball(space, position=position, color="red"))

    for position in INITIAL_BALL_POSITIONS_CM["blue"]:
        balls.append(Ball(space, position=position, color="blue"))

    render_scene(
        screen,
        wall,
        center_goal_lower,
        center_goal_upper,
        long_goal_one,
        long_goal_two,
        long_goal_one_leg_one,
        long_goal_one_leg_two,
        long_goal_two_leg_one,
        long_goal_two_leg_two,
        balls,
        red_robot,
        blue_robot,
        None,
        None,
    )

    while True:
        dt = min(clock.tick(120) / 1000, 0.05)
        requery_elapsed += dt

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return
            if event.type == pygame.MOUSEMOTION and selecting_orientation:
                orientation_mouse_world = screen_to_world(event.pos)
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                world_pos = screen_to_world(event.pos)
                if not selecting_orientation:
                    click_target_pos = world_pos
                    orientation_mouse_world = world_pos
                    selecting_orientation = True
                else:
                    dx = world_pos[0] - click_target_pos[0]
                    dy = world_pos[1] - click_target_pos[1]
                    target_angle = math.atan2(dy, dx) if math.hypot(dx, dy) > 1e-6 else red_robot.body.angle
                    pending_target_pose = (click_target_pos, target_angle)
                    selecting_orientation = False
                    click_target_pos = None
                    orientation_mouse_world = None

        if requery_elapsed >= REQUERY_INTERVAL_S:
            red_robot.stop()
            if pending_target_pose is not None:
                red_robot.set_motion_target(pending_target_pose[0], pending_target_pose[1])
            pending_target_pose = None
            requery_elapsed = 0.0

        red_robot.update(dt)
        blue_robot.update(dt)
        step_space(space, dt)

        render_scene(
            screen,
            wall,
            center_goal_lower,
            center_goal_upper,
            long_goal_one,
            long_goal_two,
            long_goal_one_leg_one,
            long_goal_one_leg_two,
            long_goal_two_leg_one,
            long_goal_two_leg_two,
            balls,
            red_robot,
            blue_robot,
            click_target_pos if selecting_orientation else None,
            orientation_mouse_world if selecting_orientation else None,
        )
