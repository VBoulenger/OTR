"""
Module to define rover class.
"""

from dataclasses import dataclass


@dataclass
class Position:
    """
    Dataclass to hold a state of the rover.
    """

    x: float  # 0 at the top border
    y: float  # 0 at the left border
    direction: float  # in deg, 0 is North.


class Rover:
    """
    A class to represent a rover with a given position on a map. The map should be a binary map.
    """

    def __init__(self, position, planet):
        self.position = position
        self.planet = planet
        self.path = self.find_path()

    def move(self, speed, rotation):
        """Move the rover with given speed and rotation"""

    def scan(self):
        """Scan the environment with its sensors"""

    def find_path(self):
        """Find path from a starting point to a goal point on the map."""
        return []

    def locate(self):
        """Find its position on the map"""
