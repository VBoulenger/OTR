"""
Define classes and simulation for a Rover that can estimate its position using a particle filter given a few landscapes.
The world is free of any obstacles, we just check that the rover is not leaving the world.
"""

import copy
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np


def gaussian(x, sigma):
    """Normal law"""
    return np.exp(-(np.power(x / sigma, 2) / 2) / (np.sqrt(2.0 * np.pi) * sigma))


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

    def __mul__(self, scalar):
        return Position(self.x * scalar, self.y * scalar, self.direction * scalar)

    def __add__(self, other):
        return Position(
            self.x + other.x, self.y + other.y, self.direction + other.direction
        )

    def __sub__(self, other):
        return Position(
            self.x - other.x, self.y - other.y, self.direction - other.direction
        )


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


class RoverParticle(BasicRover):
    """Class to represent a particle in the particle filter"""

    def __init__(self, position, planet, noise):
        super().__init__(position, planet, noise)

        # Particle filter
        self.weight = 1.0

    def __mul__(self, scalar):
        return RoverParticle(self.position_true * scalar, self.planet, self.noise)

    def __add__(self, other):
        return RoverParticle(
            self.position_true + other.position_true, self.planet, self.noise
        )

    def __sub__(self, other):
        return RoverParticle(
            self.position_true - other.position_true, self.planet, self.noise
        )

    def measurement_likelihood(self, measurements):
        """Find the likelihood of the measurement"""
        likelihood = 1.0
        for i, landmark in enumerate(self.planet.landmarks):
            dist = np.sqrt(
                (self.position_true.x - landmark.x) ** 2
                + (self.position_true.y - landmark.y) ** 2
            )
            likelihood *= gaussian(dist - measurements[i], self.noise.sensing)
        self.weight = likelihood


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
        particle_number: int = 100,
    ):
        super().__init__(position, planet, noise)

        self.position_dr: Position = (
            self.position_true
        )  # Dead reckoning position, we assume that the rover move without noise
        self.position_est: Position = (
            self.position_true
        )  # Estimated position of the rover with PF

        # Sensing
        self.observations: list[float]

        # Particle Filter

        # Initialize particle states with the initial state of the rover
        # (assumptions: it is known but that should be able to be removed
        # without causing too much troubles)

        self.particle_number = particle_number
        self.rover_particles: list[RoverParticle] = [
            RoverParticle(position, planet, Noise(1, 5.0, 5.0))
            for i in range(self.particle_number)
        ]

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

    def pf_localization(self, speed, rotation):
        """Localize rover using particle filter"""
        sum_weight = 0
        for rover in self.rover_particles:
            speed_noisy = speed + np.random.normal(0.0, self.noise.speed)
            rotation_noisy = rotation + np.random.normal(0.0, self.noise.rotation)
            rover.position_true = rover.move(
                rover.position_true, speed_noisy, rotation_noisy
            )
            rover.measurement_likelihood(self.observations)
            sum_weight += rover.weight
        self.position_est = np.sum(
            [rover * (rover.weight / sum_weight) for rover in self.rover_particles]
        ).position_true
        self.resampling()

    def resampling(self):
        """Resample with low variance"""
        indexes = []
        ind = np.random.randint(0, self.particle_number)
        beta = 0.0
        max_weight = np.max([rover.weight for rover in self.rover_particles])

        for i in range(self.particle_number):
            beta += np.random.uniform(0, 1) * 2.0 * max_weight
            while beta > self.rover_particles[ind].weight:
                beta -= self.rover_particles[ind].weight
                ind = np.mod(ind + 1, self.particle_number)
            indexes.append(ind)

        self.rover_particles = [copy.deepcopy(self.rover_particles[i]) for i in indexes]

    def time_step(self):
        """Perform a time step (i.e. move rover, sense, locate, etc.)"""
        speed = 5
        rotation = 7.5  # deg
        self.position_dr = self.move(self.position_dr, speed, rotation)

        # Add noise
        speed_noisy = speed + np.random.normal(0.0, self.noise.speed)
        rotation_noisy = rotation + np.random.normal(0.0, self.noise.rotation)
        self.position_true = self.move(self.position_true, speed_noisy, rotation_noisy)

        # Observation
        self.sense()

        # Localization
        self.pf_localization(speed, rotation)


def main():
    """Main function for the simulation"""
    # Initialization

    init_pos = Position(0, 0, 0)
    landmarks = [
        Landmark(20, 20),
        Landmark(40, 40),
        Landmark(20, 50),
        Landmark(50, 20),
        Landmark(60, 60),
        Landmark(40, 70),
        Landmark(70, 40),
    ]
    planet = Planet(100, landmarks)
    noise = Noise(0.5, 5.0, 0.5)
    rover = Rover(init_pos, planet, noise, particle_number=100)

    true_pos = [rover.position_true]
    est_pos = [rover.position_est]
    dr_pos = [rover.position_dr]

    # Simulation

    for _ in range(50):
        rover.time_step()

        true_pos.append(rover.position_true)
        est_pos.append(rover.position_est)
        dr_pos.append(rover.position_dr)

    # Results

    plt.plot([pos.x for pos in true_pos], [pos.y for pos in true_pos], label="true")
    plt.plot([pos.x for pos in est_pos], [pos.y for pos in est_pos], label="estimated")
    plt.plot(
        [pos.x for pos in dr_pos], [pos.y for pos in dr_pos], label="dead_reckoning"
    )
    plt.scatter(
        [l.x for l in rover.planet.landmarks],
        [l.y for l in rover.planet.landmarks],
        c="r",
        marker="x",
    )
    plt.xlim([0, 100])
    plt.ylim([0, 100])
    plt.legend()

    plt.show()


if __name__ == "__main__":
    main()
