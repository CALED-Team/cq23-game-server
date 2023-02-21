import pygame
import pymunk
import pymunk.pygame_util

from config import config
from game import Game
from map import Map


def run(use_pygame=False):

    m = Map(config.MAP.PATH)
    running = True
    space = pymunk.Space()
    pymunk.pygame_util.positive_y_is_up = True
    game = Game(space, m)

    if use_pygame:
        pygame.init()
        display = pygame.display.set_mode((
            m.map_width * config.GRID_SCALING,
            m.map_height * config.GRID_SCALING
        ))
        clock = pygame.time.Clock()
        draw_options = pymunk.pygame_util.DrawOptions(display)  # type: ignore

    while running:
        if use_pygame:
            # Visual mainloop
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            # Reset screen and draw all physics objects
            display.fill(pygame.Color("white"))
            space.debug_draw(draw_options)

        state = []
        for x in space.shapes:
            s = f"{x} {x.body.position} {x.body.velocity}"
            state.append(s)

        clock.get_time()

        # TODO: POST COMMUNICATIONS TO CLIENTS
        # TODO: POST REPLAY DATA
        # TODO: RECEIVE COMMUNICATIONS FROM CLIENTS - Blocking operation.

        for _ in range(config.SIMULATION.PHYSICS_ITERATIONS_PER_COMMUNICATION):

            # Update physics and do game logic.
            space.step(config.SIMULATION.PHYSICS_TIMESTEP)
            game.tick()

        if use_pygame:
            pygame.display.flip()
            clock.tick(1/config.SIMULATION.COMMUNICATION_POLLING_TIME)

    if use_pygame:
        pygame.quit()


if __name__ == "__main__":
    # setup pymunk
    space = pymunk.Space()
    use_pygame = True

    run(use_pygame)
