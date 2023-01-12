import heapq
import numpy as np
import matplotlib.pyplot as plt
import sys

class DStarLite:
    def __init__(self, start, goal, map_size):
        self.start = start
        self.goal = goal
        self.km = 0
        self.open = []
        self.closed = set()
        self.g = {self.start: float('inf')}
        self.rhs = {self.start: 0}
        self.pred = {}
        self.map_size = map_size

        self.g[self.goal] = float('inf')
        self.rhs[self.goal] = float('inf')

    def heuristic(self, a, b):
        # Manhattan distance heuristic
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def cost(self, a, b):
        # Assume constant cost for all actions
        return 1

    def update_vertex(self, u):
        if u != self.goal:
            self.rhs[u] = min([self.g[v] + self.cost(u, v) for v in self.get_successors(u)])
        if u in self.closed:
            self.open.append(u)
            heapq.heapify(self.open)
            self.closed.remove(u)

    def get_successors(self, u):
        # Example implementation - replace with actual successor function for the problem
        successors = []
        for x in range(-1, 2):
            for y in range(-1, 2):
                if x == 0 and y == 0:
                    continue
                pos = (u[0] + x, u[1] + y)
                if pos[0] < 0 or pos[1] < 0 or pos[0]>=self.map_size[0] or pos[1]>=self.map_size[1]:
                    continue
                successors.append(pos)
        if u != self.goal:
            successors.append(self.goal) # Add the goal as a possible successor
        return successors

    def compute_shortest_path(self):
        heapq.heappush(self.open, self.start)

        while (self.open and (self.g.get(self.start,float('inf')) != self.rhs.get(self.start,float('inf')) or self.rhs.get(self.goal,float('inf')) != self.g.get(self.goal,float('inf'))) or (not self.open and self.g.get(self.goal,float('inf')) != self.rhs.get(self.goal,float('inf')))):
            if not self.open:
                u = min(self.rhs.keys(), key=lambda x: self.rhs[x] + self.heuristic(x, self.start))
                self.g[u] = self.rhs[u]
            else:
                u = heapq.heappop(self.open)
                if self.g[u] > self.rhs[u]:
                    self.g[u] = self.rhs[u]
                    self.closed.add(u)
                else:
                    self.g[u] = float('inf')
                    self.update_vertex(u)
                    self.closed.add(u)
            for v in self.get_successors(u):
                if v in self.closed:
                    continue
                if self.g.get(v,float('inf')) != self.rhs.get(v,float('inf')):
                    self.pred[v] = u
                    self.update_vertex(v)


if __name__ == '__main__':
    # Map size
    map_size = (100, 100)

    # Generate random map
    map = np.random.randint(0, 2, map_size)

    # Start and goal positions
    start = (0, 0)
    goal = (map_size[0] - 1, map_size[1] - 1)
    if start[0]<0 or start[1]<0 or start[0]>=map_size[0] or start[1]>=map_size[1] or goal[0]<0 or goal[1]<0 or goal[0]>=map_size[0] or goal[1]>=map_size[1]:
        print("Start or goal positions are out of the map boundaries")
        sys.exit()
    # Create DStarLite object and compute shortest path
    dsl = DStarLite(start, goal, map_size)
    dsl.compute_shortest_path()

    # Get the path from start to goal
    path = [goal]
    current = goal
    while current != start:
        current = dsl.pred.get(current,None)
        if current is None:
            print("There is no path from start to goal")
            sys.exit()

        path.append(current)
    path.reverse()

    # Plot the map and the path
    plt.imshow(map, cmap='gray', origin='lower', extent=(0, map_size[0], 0, map_size[1]))
    plt.scatter(*zip(*path), color='r')
    plt.show()

