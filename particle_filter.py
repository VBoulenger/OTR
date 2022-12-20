"""
Define classes and simulation for a Rover that can estimate its position using a particle filter given a few landscapes.
The world is free of any obstacles, we just check that the rover is not leaving the world.
"""

import copy
import heapq
from dataclasses import dataclass

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("TkAgg")


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

    maze: np.array
    landmarks: list[Landmark]

    def size_x(self):
        """Return the size of the world along the x axis"""
        return self.maze.shape[0]

    def size_y(self):
        """Return the size of the world along the y axis"""
        return self.maze.shape[1]


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

        # Check for collisions with world boundaries
        if next_pos.x < 0:
            next_pos.x = 0
        elif next_pos.x > self.planet.size_x() - 1:
            next_pos.x = self.planet.size_x() - 1
        if next_pos.y < 0:
            next_pos.y = 0
        elif next_pos.y > self.planet.size_y() - 1:
            next_pos.y = self.planet.size_y() - 1

        # Check for collisions with walls
        if self.planet.maze[round(next_pos.y)][round(next_pos.x)] == 1:
            if self.planet.maze[round(next_pos.y)][round(position.x)] == 1:
                next_pos.y = position.y
            if self.planet.maze[round(position.y)][round(next_pos.x)] == 1:
                next_pos.x = position.x
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
        goal: tuple[int],
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

        # Path Planning: A*
        self.path = self.find_path((self.position_est.x, self.position_est.y), goal)

        # Sensing: RFID/LIDAR to determined landmarks ?
        self.observations: list[float]

        # Localization: Particle Filter

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
        position_est = np.sum(
            [rover * (rover.weight / sum_weight) for rover in self.rover_particles]
        ).position_true
        self.resampling()
        return position_est

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

    def path_heuristic(self, pos1, pos2):
        """Heuristic used in A* algorithm"""
        return np.abs(pos2[0] - pos1[0]) + np.abs(pos2[1] - pos1[1])

    def find_path(self, start, goal):
        """Find path from a starting point to a goal point on the map using A* algorithm"""

        neighbors = [
            (0, 1),
            (0, -1),
            (1, 0),
            (-1, 0),
            (1, 1),
            (1, -1),
            (-1, 1),
            (-1, -1),
        ]

        close_set = set()
        came_from = {}
        gscore = {start: 0}
        fscore = {start: self.path_heuristic(start, goal)}
        oheap = []

        heapq.heappush(oheap, (fscore[start], start))
        while oheap:
            current = heapq.heappop(oheap)[1]

            if current == goal:
                data = []
                while current in came_from:
                    data.append(current)
                    current = came_from[current]
                return data[::-1]

            close_set.add(current)
            for i, j in neighbors:
                neighbor = current[0] + i, current[1] + j
                tentative_g_score = gscore[current] + self.path_heuristic(
                    current, neighbor
                )
                if (
                    neighbor[0] < 0 or neighbor[0] > self.planet.size_x() - 1
                ):  # TODO check order of x and y on size when using rectangular map !!!!
                    continue

                if neighbor[1] < 0 or neighbor[1] > self.planet.size_y() - 1:
                    continue

                if self.planet.maze[neighbor[0]][neighbor[1]] == 1:
                    continue

                if neighbor in close_set and tentative_g_score >= gscore.get(
                    neighbor, 0
                ):
                    continue

                if tentative_g_score < gscore.get(neighbor, 0) or neighbor not in [
                    i[1] for i in oheap
                ]:
                    came_from[neighbor] = current
                    gscore[neighbor] = tentative_g_score
                    fscore[neighbor] = tentative_g_score + self.path_heuristic(
                        neighbor, goal
                    )
                    heapq.heappush(oheap, (fscore[neighbor], neighbor))
        return []

    def calc_control(self):
        """The rover compute its control input for the next step in order to reach its next waypoint."""
        if self.path:
            print(f"Moving towards {self.path[0]}")
            delta_x = self.path[0][1] - self.position_est.x
            delta_y = self.path[0][0] - self.position_est.y
            speed = np.sqrt(delta_x**2 + delta_y**2)
            angle_with_horizontal_axis = np.arctan2(delta_y, delta_x) * 180 / np.pi
            rotation = angle_with_horizontal_axis - self.position_est.direction
            return speed, rotation
        return 0, 0

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
        self.position_est = self.pf_localization(speed, rotation)


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

    # Load maze

    img_path = "images/map_200.png"
    maze = cv2.bitwise_not(cv2.imread(img_path, 0)) / 255.0
    maze[maze != 0] = 1
    planet = Planet(maze, landmarks)
    noise = Noise(0.5, 5.0, 0.5)
    final_pos = (199, 199)
    rover = Rover(init_pos, final_pos, planet, noise)

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

    plt.imshow(maze, cmap="gray_r")
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
    plt.xlim([0, planet.size_x() - 1])
    plt.ylim([0, planet.size_y() - 1])
    plt.legend()

    plt.show()


if __name__ == "__main__":
    main()
