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

    def __init__(self, position, planet, goal):
        self.position = position
        self.planet = planet
        self.path = self.find_path((self.position.x, self.position.y), goal)

    def move(self, speed, rotation):
        """Move the rover with given speed and rotation"""

    def scan(self):
        """Scan the environment with its sensors"""

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
