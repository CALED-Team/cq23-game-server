import pygame
import pymunk
import pymunk.pygame_util

from config import config
from game import Game
from map import Map


def run_pygame():
    # PyGame init
    pygame.init()
    m = Map(config.MAP.PATH)
    display = pygame.display.set_mode(
        (m.map_width * config.GRID_SCALING, m.map_height * config.GRID_SCALING)
    )
    clock = pygame.time.Clock()
    FPS = 50
    running = True
    # Physics stuff
    space = pymunk.Space()
    pymunk.pygame_util.positive_y_is_up = True
    draw_options = pymunk.pygame_util.DrawOptions(display)  # type: ignore

    # example objects
    game = Game(space, m)

    while running:
        # Did the user click the window close button?
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # Fill the background with white
        display.fill(pygame.Color("white"))

        # Draw stuff
        space.debug_draw(draw_options)

        state = []
        for x in space.shapes:
            s = f"{x} {x.body.position} {x.body.velocity}"
            state.append(s)

        # Update physics
        space.step(1 / FPS)
        game.tick()

        pygame.display.flip()
        clock.tick(FPS)

    # Done! Time to quit.
    pygame.quit()


def run_pymunk():
    FPS = 50
    running = True

    # Physics stuff
    space = pymunk.Space()

    # example objects
    map = Map(config.MAP.PATH)
    game = Game(space, map)

    while running:
        state = []
        for x in space.shapes:
            if x.collision_type == config.COLLISION_TYPE.TANK:
                s = f"{x} {x.body.position} {x.body.velocity}"
                state.append(s)

        # Update physics
        space.step(1 / FPS)
        game.tick()


if __name__ == "__main__":
    # setup pymunk
    space = pymunk.Space()
    visualise_game = True

    if visualise_game:  # run game in pygame
        run_pygame()
    else:
        run_pymunk()
