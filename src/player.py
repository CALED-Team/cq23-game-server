from collections import deque

from gameObjects.tank import Tank
from map import Map


class Player:
    def __init__(self, player_object: Tank, map: Map):
        self.gameobject: Tank = player_object
        self.map: Map = map
        self.path: deque = deque()
        self.target: tuple[int, int] | None = None

        # TODO: setup communication channel with the player's bot

    def tick(self):
        """This will be called at every tick."""
        self._traverse_path()

    def request_move(self):
        # supply the relevant gamestate info to the player, then await their move
        pass

    def _traverse_path(self):
        """Move the player through a previously calculated path (self.path) until the path is complete"""
        if not self.path:
            self._set_direction(
                (0, 0)
            )  # stop moving the player once the target has been reached
            return
        if (
            sum(
                map(
                    lambda x, y: abs(x - y), self.gameobject.body.position, self.path[0]
                )
            )
            <= 0.3
        ):  # pymunk coordinates are floating point nums
            self.path.popleft()
            return
        self.gameobject.move_to_pos(self.path[0])

    def _set_path(self, coord: tuple[int, int]):
        """calculate and set the path attribute if the target is coord

        Args:
            coord (tuple[int, int]): the target coordinate the function will create a path to
        """
        self.path = deque(
            map(
                lambda p: self.map.to_global_coords(*p),
                self.map.path(
                    self.map.from_global_coords(*self.gameobject.body.position),
                    self.map.from_global_coords(*coord),
                ),
            )
        )
        self.target = coord

    def _set_direction(self, direction: tuple[int, int]):
        """manually set the direction that the tank should move towards

        Args:
            direction (tuple[int, int]): The direction the tank should move towards (i.e. (x,y)). (1,1) is up and right
        """
        self.gameobject.set_velocity(direction)
