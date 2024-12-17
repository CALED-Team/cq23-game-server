from __future__ import annotations

import logging
import random
from collections import defaultdict
from collections.abc import Callable

import pymunk

from communicator import Communicator
from config import config
from gameObjects.boundary import Boundary
from gameObjects.bullet import Bullet
from gameObjects.powerup import Powerup, PowerupType
from gameObjects.tank import Tank
from gameObjects.wall import Wall
from log import log_with_time
from map import Map
from player import Player
from replay import ReplayManager


class Game:
    def __init__(self, space: pymunk.Space, map: Map, replay_manager: ReplayManager):
        self.space = space
        self.map = map
        self.game_objects = list(self.map.create_game_objects(self.space))
        self.replay_manager = replay_manager

        self.comms = Communicator()
        tanks = filter(lambda go: isinstance(go, Tank), self.game_objects)
        self.players = {
            client_info["id"]: Player(tank, map, client_info)
            for tank, client_info in zip(tanks, self.comms.client_info)
        }
        self.path_indicators = {
            client_info["id"]: [] for client_info in self.comms.client_info
        }

        # create map boundary
        self.boundary = Boundary(
            self.space,
            self.map.to_global_coords(-0.5, -0.5),
            self.map.to_global_coords(
                self.map.map_height - 0.5, self.map.map_width - 0.5
            ),
        )
        self.closing_boundary = Boundary(
            self.space,
            self.map.to_global_coords(-0.5, -0.5),
            self.map.to_global_coords(
                self.map.map_height - 0.5, self.map.map_width - 0.5
            ),
            is_closing_boundary=True,
        )
        self.game_objects.append(self.boundary)
        self.game_objects.append(self.closing_boundary)

        self.tick_count = 0

        self.add_collision_handlers()
        self.add_separate_handlers()
        self.remove_collision_handlers()

        self.send_map_info_to_clients()

    def send_map_info_to_clients(self):
        log_with_time("Sending map info to clients")
        self.replay_manager.set_game_info(self.space)
        comms_line = self.replay_manager.sync_object_updates_in_comms()
        comms_line["code"] = "KH73TU"
        self.comms.post_init_world_message(comms_line)
        self.comms.terminate_init_world_sequence()

    def add_separate_handlers(self):
        collision_groups: list[tuple[int, int, Callable | None]] = [
            (
                config.COLLISION_TYPE.BULLET,
                config.COLLISION_TYPE.CLOSING_BOUNDARY,
                self.bullet_boundary_separate_handler,
            )
        ]

        for coltype_a, coltype_b, handler in collision_groups:
            collision_handler = self.space.add_collision_handler(coltype_a, coltype_b)
            if handler:
                collision_handler.separate = handler

    def bullet_boundary_separate_handler(
        self, arbiter: pymunk.Arbiter, space: pymunk.Space, data
    ):
        for shape in arbiter.shapes:
            if isinstance(shape._gameobject, Boundary):
                boundary_shape = shape
            if isinstance(shape._gameobject, Bullet):
                bullet_shape = shape

        if bullet_shape is not None and boundary_shape is not None:
            bullet_shape.body.velocity = list(
                map(
                    lambda a, b: a - 2 * b,
                    bullet_shape.body.velocity,
                    boundary_shape.body.velocity,
                )
            )

    def add_collision_handlers(self):
        """register all collision handlers in the pymunk space"""
        collision_groups: list[tuple[int, int, Callable | None]] = [
            (config.COLLISION_TYPE.TANK, config.COLLISION_TYPE.WALL, None),
            (config.COLLISION_TYPE.TANK, config.COLLISION_TYPE.DESTRUCTIBLE_WALL, None),
            (config.COLLISION_TYPE.TANK, config.COLLISION_TYPE.TANK, None),
            (config.COLLISION_TYPE.BOUNDARY, config.COLLISION_TYPE.TANK, None),
            (config.COLLISION_TYPE.BOUNDARY, config.COLLISION_TYPE.BULLET, None),
            (
                config.COLLISION_TYPE.CLOSING_BOUNDARY,
                config.COLLISION_TYPE.BULLET,
                lambda a, b, c: self.bullet_collision_handler(a, b, c, bouncy=True),
            ),
            (
                config.COLLISION_TYPE.BULLET,
                config.COLLISION_TYPE.TANK,
                lambda a, b, c: self.bullet_collision_handler(a, b, c, bouncy=False),
            ),
            (
                config.COLLISION_TYPE.BULLET,
                config.COLLISION_TYPE.BULLET,
                lambda a, b, c: self.bullet_collision_handler(a, b, c, bouncy=False),
            ),
            (
                config.COLLISION_TYPE.BULLET,
                config.COLLISION_TYPE.DESTRUCTIBLE_WALL,
                lambda a, b, c: self.bullet_collision_handler(a, b, c, bouncy=False),
            ),
            (
                config.COLLISION_TYPE.BULLET,
                config.COLLISION_TYPE.WALL,
                lambda a, b, c: self.bullet_collision_handler(a, b, c, bouncy=True),
            ),
            (
                config.COLLISION_TYPE.CLOSING_BOUNDARY,
                config.COLLISION_TYPE.TANK,
                self.closing_boundary_collision_handler,
            ),
            (
                config.COLLISION_TYPE.TANK,
                config.COLLISION_TYPE.POWERUP,
                self.powerup_collision_handler,
            ),
        ]

        for coltype_a, coltype_b, handler in collision_groups:
            collision_handler = self.space.add_collision_handler(coltype_a, coltype_b)
            if handler:
                collision_handler.post_solve = handler

    def remove_collision_handlers(self):
        collision_groups: list[tuple[int, int]] = [
            (config.COLLISION_TYPE.BULLET, config.COLLISION_TYPE.POWERUP),
            (config.COLLISION_TYPE.TANK, config.COLLISION_TYPE.PATH),
            (config.COLLISION_TYPE.BULLET, config.COLLISION_TYPE.PATH),
            (config.COLLISION_TYPE.WALL, config.COLLISION_TYPE.PATH),
            (config.COLLISION_TYPE.DESTRUCTIBLE_WALL, config.COLLISION_TYPE.PATH),
            (config.COLLISION_TYPE.BOUNDARY, config.COLLISION_TYPE.PATH),
            (config.COLLISION_TYPE.CLOSING_BOUNDARY, config.COLLISION_TYPE.PATH),
            (config.COLLISION_TYPE.POWERUP, config.COLLISION_TYPE.PATH),
        ]
        for col_type_a, col_type_b in collision_groups:
            poweup_collision_handler = self.space.add_collision_handler(
                col_type_a, col_type_b
            )
            poweup_collision_handler.begin = lambda a, b, c: False

    def powerup_collision_handler(
        self, arbiter: pymunk.Arbiter, space: pymunk.Space, data
    ):
        """collision handler for collisions between tanks and powerups

        Args:
            arbiter (pymunk.Arbiter): pymunk provided arg
            space (pymunk.Space): pymunk provided arg
            data (_type_): pymunk provided arg
        """
        for shape in arbiter.shapes:
            if isinstance(shape._gameobject, Powerup):
                powerup: Powerup = shape._gameobject
                space.remove(shape, shape.body)
                self.game_objects.remove(
                    shape._gameobject
                )  # remove reference to game object

        for shape in arbiter.shapes:
            if isinstance(shape._gameobject, Tank):
                shape._gameobject.apply_powerup(powerup)
                self.replay_manager.record_deleted_object(powerup.id)

    def register_deleted_object(self, shape: pymunk.Shape):
        # record destruction in replay file
        if shape.collision_type in [
            config.COLLISION_TYPE.BULLET,
            config.COLLISION_TYPE.DESTRUCTIBLE_WALL,
            config.COLLISION_TYPE.TANK,
        ]:
            self.replay_manager.record_deleted_object(shape._gameobject.id)

    def closing_boundary_collision_handler(
        self, arbiter: pymunk.Arbiter, space: pymunk.Space, data
    ):
        """collision handler for collisions with the closing boundary

        Args:
            arbiter (pymunk.Arbiter): pymunk provided arg
            space (pymunk.Space): pymunk provided arg
            data (_type_): pymunk provided arg
        """
        for shape in arbiter.shapes:
            if shape._gameobject.apply_damage(
                config.CLOSING_BOUNDARY.DAMAGE
            ).is_destroyed():
                space.remove(shape, shape.body)
                self.game_objects.remove(
                    shape._gameobject
                )  # remove reference to game object

                self.register_deleted_object(shape)

    def bullet_collision_handler(
        self, arbiter: pymunk.Arbiter, space: pymunk.Space, data, bouncy
    ):
        """collision handler for collisions that would cause HP loss to an object caused by a bullet

        Args:
            arbiter (pymunk.Arbiter): pymunk provided arg
            space (pymunk.Space): pymunk provided arg
            data (_type_): pymunk provided arg
        """
        damage = config.BULLET.DAMAGE

        for shape in arbiter.shapes:
            if isinstance(shape._gameobject, Bullet):
                damage = shape._gameobject.damage

        for shape in arbiter.shapes:
            if shape._gameobject.apply_damage(damage).is_destroyed() or (
                not bouncy and isinstance(shape._gameobject, Bullet)
            ):
                if shape.collision_type == config.COLLISION_TYPE.DESTRUCTIBLE_WALL:
                    self.map.register_wall_broken(shape.body.position)
                space.remove(shape, shape.body)
                try:
                    self.game_objects.remove(
                        shape._gameobject
                    )  # remove reference to game object
                except ValueError:
                    logging.warning(
                        f"Could not find {shape._gameobject.id} in self.game_object"
                    )

                self.register_deleted_object(shape)

    def handle_client_response(self):
        message = self.comms.get_message()
        for client_id in message:
            self.game_objects.extend(  # keep the reference to any object created
                self.players[client_id].register_actions(actions=message[client_id])
            )
            # show the path on the map:
            if (
                message[client_id] is not None
                and "path" in message[client_id]
                and "move" not in message[client_id]
            ):
                self.remove_path_indicators(client_id)
                for p in self.players[client_id].action["path"]:
                    body = pymunk.Body(body_type=pymunk.Body.STATIC)
                    body.position = p
                    shape = pymunk.Circle(body=body, radius=5)
                    shape.collision_type = config.COLLISION_TYPE.PATH
                    self.path_indicators[client_id].append((body, shape))
                    self.space.add(body, shape)
            elif (
                message[client_id] is not None
                and "path" not in message[client_id]
                and "move" in message[client_id]
            ):
                self.remove_path_indicators(client_id)

    def remove_path_indicators(self, client_id):
        for b, s in self.path_indicators[client_id]:
            self.space.remove(b, s)
        self.path_indicators[client_id] = []

    def tick(self):
        """called at every tick"""
        self.tick_count += 1
        self._play_turn()
        if self.tick_count % config.TICKS_PER_POWERUP == 0:
            self.spawn_powerup()
        if (
            self.tick_count
            % (6000 * (config.GRID_SCALING / config.CLOSING_BOUNDARY.VELOCITY))
            == 0
        ):
            self.map.update_traversability_boundary()
        if self._is_terminal():
            self.comms.terminate_game()
            return True
        return False

    def spawn_powerup(self) -> Powerup:
        boundary_vertices = self.closing_boundary.get_vertices()
        xrange = (
            min(map(lambda x: int(x[0]), boundary_vertices)),
            max(map(lambda x: int(x[0]), boundary_vertices)),
        )
        yrange = (
            min(map(lambda x: int(x[1]), boundary_vertices)),
            max(map(lambda x: int(x[1]), boundary_vertices)),
        )

        for _ in range(15):  # 15 retries to find position to plant powerup
            collision_detected = False
            poweup_coord = (random.randint(*xrange), random.randint(*yrange))
            powerup_type = random.choice(list(PowerupType))

            powerup = Powerup(
                space=self.space,
                coord=poweup_coord,
                powerup_type=powerup_type,
            )
            for object in self.game_objects:
                if (isinstance(object, Wall) or isinstance(object, Tank)) and len(
                    object.shape.shapes_collide(powerup.shape).points
                ) > 0:
                    collision_detected = True
                    self.space.remove(powerup.shape, powerup.body)
                    del powerup
                    break
            if not collision_detected:
                self.game_objects.append(powerup)
                return powerup

    def _play_turn(self) -> None:
        for player in self.players.values():
            player.tick()

    def _is_terminal(self) -> bool:
        active_players = 0
        for player in self.players.values():
            if player.gameobject.hp > 0:
                active_players += 1
        if active_players > 1:
            return False
        return True

    def results(self) -> defaultdict:
        results = defaultdict(list)
        for playerId, player in self.players.items():
            if player.gameobject.hp > 0:
                results["victor"].append(playerId)
            else:
                results["vanquished"].append(playerId)
        return results
