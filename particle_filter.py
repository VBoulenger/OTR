"""
Define classes and simulation for a Rover that can estimate its position using a particle filter given a few landscapes.
The world is free of any obstacles, we just check that the rover is not leaving the world.
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class Landmark:
    """Represents a landmark at a given position"""

    x: float
    y: float


@dataclass
class Planet:
    """Represents a planet (world) with a given size and landmarks"""

    size: float
    landmarks: list[Landmark]


@dataclass
class Position:
    """Represents a state of the rover"""

    x: float
    y: float
    direction: float


@dataclass
class Noise:
    """Dataclass to hold noise information"""

    sensing: float
    rotation: float
    speed: float


class BasicRover:
    """Define class to represent a basic rover. It can move on its planet and get information from the landmarks"""

    def __init__(self, position: Position, planet: Planet, noise: Noise):
        self.position_true: Position = position
        self.planet: Planet = planet
        self.noise: Noise = noise

    def move(self, position, speed, rotation):
        """Move the rover with given speed and rotation"""

        next_dir = np.mod(position.direction + rotation, 360)

        next_pos = Position(position.x, position.y, next_dir)

        next_pos.x += np.cos(next_pos.direction * np.pi / 180) * speed
        next_pos.y += np.sin(next_pos.direction * np.pi / 180) * speed

        if next_pos.x < 0:
            next_pos.x = 0
        elif next_pos.x > self.planet.size - 1:
            next_pos.x = self.planet.size - 1
        if next_pos.y < 0:
            next_pos.y = 0
        elif next_pos.y > self.planet.size - 1:
            next_pos.y = self.planet.size - 1

        return next_pos


class Rover(BasicRover):
    """
    Class to represent the actual rover. It inherits from BasicRover but also has
    some additional methods to locate itself and determine which path it should take.
    """

    def __init__(
        self,
        position: Position,
        planet: Planet,
        noise: Noise,
    ):
        super().__init__(position, planet, noise)

        self.position_dr: Position = self.position_true
        self.position_est: Position = self.position_true

        # Sensing
        self.observations: list[float]

    def sense(self):
        """Get distance between the rover and the landmarks"""
        self.observations = []
        for landmark in self.planet.landmarks:
            dist = np.sqrt(
                (self.position_true.x - landmark.x) ** 2
                + (self.position_true.y - landmark.y) ** 2
            )
            dist += np.random.normal(0.0, self.noise.sensing)
            self.observations.append(dist)
