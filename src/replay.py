from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, is_dataclass
from enum import Enum
from typing import Any

from gameObjects.powerup import PowerupType


class EventType(str, Enum):
    BULLET_SPAWN = "BULLET_SPAWN"
    BULLET_DESTROYED = "BULLET_DESTROYED"
    TANK_HEALTH_LOSS = "TANK_HEALTH_LOSS"
    TANK_DESTROYED = "TANK_DESTROYED"
    POWERUP_SPAWN = "POWERUP_SPAWN"
    WALL_HEALTH_LOSS = "WALL_HEALTH_LOSS"
    WALL_DESTROYED = "WALL_DESTROYED"
    POWERUP_COLLECTED = "POWERUP_COLLECTED"


@dataclass
class Event:
    """
    Single use event data to be used in replays.
    Data should be a json-serializable object
    """

    event_type: EventType
    data: dict = field(default_factory=dict)

    @classmethod
    def tank_health_loss(cls, tank_id: str, position: tuple[float, float]):
        return Event(EventType.TANK_HEALTH_LOSS, {"id": tank_id, "position": position})

    @classmethod
    def tank_destroyed(cls, tank_id: str, position: tuple[float, float]):
        return Event(EventType.TANK_DESTROYED, {"id": tank_id, "position": position})

    @classmethod
    def wall_destroyed(cls, wall_id: str, position: tuple[float, float]):
        return Event(EventType.WALL_DESTROYED, {"id": wall_id, "position": position})

    @classmethod
    def wall_health_loss(cls, wall_id: str, position: tuple[float, float]):
        return Event(EventType.WALL_HEALTH_LOSS, {"id": wall_id, "position": position})

    @classmethod
    def bullet_destroyed(cls, bullet_id: str, position: tuple[float, float]):
        return Event(
            EventType.BULLET_DESTROYED, {"id": bullet_id, "position": position}
        )

    @classmethod
    def bullet_spawn(
        cls,
        bullet_id: str,
        tank_id: str,
        position: tuple[float, float],
        velocity: tuple[float, float],
        angle: float,
    ):
        return Event(
            EventType.BULLET_SPAWN,
            {
                "id": bullet_id,
                "tank_id": tank_id,
                "position": position,
                "velocity": velocity,
                "angle": angle,
            },
        )

    @classmethod
    def powerup_collected(
        cls,
        tank_id: str,
        position: tuple[float, float],
        powerup_id: str,
        powerup_type: PowerupType,
    ):
        return Event(
            EventType.POWERUP_COLLECTED,
            {
                "tank_id": tank_id,
                "position": position,
                "powerup_id": powerup_id,
                "powerup_type": powerup_type,
            },
        )

    @classmethod
    def powerup_spawn(
        cls, powerup_id: str, position: tuple[float, float], powerup_type: PowerupType
    ):
        return Event(
            EventType.POWERUP_SPAWN,
            {
                "position": position,
                "powerup_id": powerup_id,
                "powerup_type": powerup_type,
            },
        )


class ReplayJSONEncoder(json.JSONEncoder):
    def default(self, o):
        if is_dataclass(o):
            return asdict(o)
        elif isinstance(o, float):
            return f"{o:.4f}"
        return super().default(o)


class ReplayManager:
    """
    Manager for posting replays to a file.
    Replay format is a file where each line contains a JSON object.
    Each json object is either a custom message, or a tick update, containing events and updated information.

    Be sure to execute instance.close() once you are certain no more replays will be passed through.
    """

    def __init__(self, output_path: str) -> None:
        self.output_path = output_path
        self._file = open(output_path, "w")
        self.events = []
        self.current_info = {}
        self.new_info = {}
        self.comms_info = {}

    def add_event(self, e: Event) -> None:
        self.events.append(e)

    def set_info(self, key: str, data: dict[str, Any]) -> None:
        self.new_info[key] = data

    def post_custom_replay_line(self, obj: dict) -> None:
        """
        Post a custom JSON message to replay.
        Used for initial messaging / finale
        """
        self._file.write(
            json.dumps(obj, cls=ReplayJSONEncoder, separators=(",", ":")) + "\n"
        )

    def set_game_info(self, space):
        for x in space.shapes:
            self.set_info(
                x._gameobject.id,
                x._gameobject.info(),
            )

    def post_replay_line(self, include_events=False) -> dict:
        updated_info = {}
        for key in self.new_info:
            if self.current_info.get(key, None) != self.new_info[key]:
                updated_info[key] = self.new_info[key]

        obj = {
            "events": list(map(asdict, self.events)) if include_events else [],
            "object_info": updated_info,
        }
        self._file.write(
            json.dumps(obj, cls=ReplayJSONEncoder, separators=(",", ":")) + "\n"
        )

        # Update stale locations
        self.current_info.update(updated_info)
        self.new_info = {}

    def get_comms_line(self):
        comms_updated_info = {}
        for key in self.new_info:
            if self.comms_info.get(key, None) != self.new_info[key]:
                comms_updated_info[key] = self.new_info[key]
        comms_obj = {
            "events": list(map(asdict, self.events)),
            "object_info": comms_updated_info,
        }

        # Clear events and update all objects' info
        self.comms_info.update(comms_updated_info)
        self.events = []
        self.new_info = {}

        return comms_obj

    def close(self):
        self._file.close()
