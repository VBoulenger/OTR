"""
Module to define rover class.
"""

import heapq
from dataclasses import dataclass

import numpy as np


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

    # FIXME: Discrete world
    # Un problème se pose: on travaille dans un monde discret et c'est pour
    # l'instant incompatible avec les erreurs de vitesse et de rotation et la
    # localisation. Il faudrait réfléchir pour changer ça. Une idée: voir le
    # path trouvé par l'algo de pathfinding comme une suite de waypoints que le
    # robot doit suivre ? À creuser.

    def __init__(self, position, planet, goal):
        self.position = position
        self.planet = planet
        self.path = self.find_path((self.position.x, self.position.y), goal)

    def autoguidance_step(self):
        """Iterate this function to move the rover along its computed path"""
        if self.path:
            print(f"Moving to {self.path[0]}")
            delta_x = self.path[0][0] - self.position.x
            delta_y = self.path[0][1] - self.position.y
            speed = np.sqrt(delta_x**2 + delta_y**2)
            angle_with_horizontal_axis = np.arctan2(delta_y, -delta_x) * 180 / np.pi
            rotation = 360 - angle_with_horizontal_axis - self.position.direction
            self.move(speed, rotation)
            self.path.pop(0)

    def move(self, speed, rotation):
        """Move the rover with given speed and rotation"""
        next_pos = Position(self.position.x, self.position.y, self.position.direction)
        next_pos.direction = (self.position.direction + rotation) % 360
        if next_pos.direction < 0:
            next_pos.direction += 360
        next_pos.x -= np.cos(next_pos.direction * np.pi / 180) * speed
        next_pos.y -= np.sin(next_pos.direction * np.pi / 180) * speed
        if next_pos.x < 0:
            next_pos.x = 0
        elif next_pos.x > self.planet.shape[0] - 1:
            next_pos.x = self.planet.shape[0] - 1
        if next_pos.y < 0:
            next_pos.y = 0
        elif next_pos.y > self.planet.shape[1] - 1:
            next_pos.y = self.planet.shape[1] - 1
        if self.planet[round(next_pos.x)][round(next_pos.y)] == 1:
            if self.planet[round(self.position.x)][round(next_pos.y)] == 1:
                next_pos.y = self.position.y
            if self.planet[round(next_pos.x)][round(self.position.y)] == 1:
                next_pos.x = self.position.x
        self.position = next_pos

    def scan(self):
        """Scan the environment with its sensors"""
        # TODO

    def path_heuristic(self, pos1, pos2):
        """Heuristic used in A* algorithm"""
        return np.abs(pos2[0] - pos1[0]) + np.abs(pos2[1] - pos1[1])

    def find_path(self, start, goal):
        """Find path from a starting point to a goal point on the map using A* algorithm"""
        # FIXME: Add DstarLite ? To add discoverable map

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
                if neighbor[0] < 0 or neighbor[0] > self.planet.shape[0] - 1:
                    continue

                if neighbor[1] < 0 or neighbor[1] > self.planet.shape[1] - 1:
                    continue

                if self.planet[neighbor[0]][neighbor[1]] == 1:
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

    def locate(self):
        """Find its position on the map"""
        # TODO implement EKF or PF
