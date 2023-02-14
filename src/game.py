import logging
from collections.abc import Callable

import pymunk

from config import config
from gameObjects.tank import Tank
from map import Map
from player import Player


class Game:
    def __init__(self, space: pymunk.Space, map: Map):
        self.space = space
        self.map = map
        self.game_objects = list(self.map.create_game_objects(self.space))

        tanks = filter(lambda go: isinstance(go, Tank), self.game_objects)
        self.players = [Player(tank, map) for tank in tanks]

        for player in self.players:  # TODO: remove this. It's only for testing purposes
            player._set_path((150, 200))

        self.add_collision_handlers()

    def add_collision_handlers(self):
        collision_groups: list[tuple[int, int, Callable | None]] = [
            (config.COLLISION_TYPE.TANK, config.COLLISION_TYPE.WALL, None),
            (config.COLLISION_TYPE.TANK, config.COLLISION_TYPE.DESTRUCTIBLE_WALL, None),
            (config.COLLISION_TYPE.TANK, config.COLLISION_TYPE.TANK, None),
            (
                config.COLLISION_TYPE.BULLET,
                config.COLLISION_TYPE.TANK,
                self.damage_collision_handler,
            ),
            (
                config.COLLISION_TYPE.BULLET,
                config.COLLISION_TYPE.BULLET,
                self.damage_collision_handler,
            ),
            (
                config.COLLISION_TYPE.BULLET,
                config.COLLISION_TYPE.DESTRUCTIBLE_WALL,
                self.damage_collision_handler,
            ),
            (
                config.COLLISION_TYPE.BULLET,
                config.COLLISION_TYPE.WALL,
                self.damage_collision_handler,
            ),
        ]

        for coltype_a, coltype_b, handler in collision_groups:
            collision_handler = self.space.add_collision_handler(coltype_a, coltype_b)
            if handler:
                collision_handler.post_solve = handler

    def damage_collision_handler(
        self, arbiter: pymunk.Arbiter, space: pymunk.Space, data
    ):
        for go in self.game_objects[:]:
            for shape in arbiter.shapes:
                if shape == go.shape:
                    if go.apply_damage(config.BULLET.DAMAGE).is_destroyed():
                        # TODO: remove the destroyed object from the map
                        if (
                            shape.collision_type
                            == config.COLLISION_TYPE.DESTRUCTIBLE_WALL
                        ):
                            self.map.register_wall_broken(shape._wall_coords)
                        self.space.remove(shape, shape.body)
                        self.game_objects.remove(go)  # remove reference to game object

    def tick(self):
        self._play_turn()

    def _play_turn(self):
        for player in self.players:
            player.tick()

    def _is_terminal(self):
        # check which players still have hp
        # if both players have 0 hp, it is a draw. Else the player with 0 hp loses
        pass

    def run(self):
        # initialise state
        logging.info("Starting game between players: ... and ...")
        # while true (or number of turns is less than max turns), play_move
        pass
