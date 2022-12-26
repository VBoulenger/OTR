"""
Define classes and simulation for a Rover that can estimate its position using a particle filter given a few landscapes.
The world is free of any obstacles, we just check that the rover is not leaving the world.
"""

import copy
import heapq
from dataclasses import dataclass
import math
import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation

matplotlib.use("TkAgg")

# Warning: this should be changed to your local install of FFMPEG if you want to save the matplotlib animation.
plt.rcParams["animation.ffmpeg_path"] = "/usr/bin/ffmpeg"

#--------------------------------------------------
Q = np.diag([
    0.1,  # variance of location on x-axis
    0.1,  # variance of location on y-axis
    np.deg2rad(1.0),  # variance of yaw angle
    1.0  # variance of velocity
]) ** 2  # predict state covariance
R = np.diag([1.0, 1.0]) ** 2  # Observation x,y position covariance
A_k_minus_1 = np.array([[1.0,  0,   0],
                        [  0,1.0,   0],
                        [  0,  0, 1.0]])

process_noise_v_k_minus_1 = np.array([0.01,0.01,0.003])

# State model noise covariance matrix Q_k
Q_k = np.array([[1.0,   0,   0],
                [  0, 1.0,   0],
                [  0,   0, 1.0]])

# Measurement matrix H_k
H_k = np.array([[1.0,  0,   0],
                [  0,1.0,   0],
                [  0,  0, 1.0]])

# Sensor measurement noise covariance matrix R_k
R_k = np.array([[1.0,   0,    0],
                [  0, 1.0,    0],
                [  0,    0, 1.0]])

# Sensor Noise Vector
sensor_noise_w_k = np.array([0.07,0.07,0.04])

#--------------------------------------------------



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
    def to_array(self,speed,rotation):
        return np.array([[self.x],[self.y],[rotation],[speed]])
 

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
        max_speed: float = 2,
        particle_number: int = 100,
    ):
        super().__init__(position, planet, noise)

        self.position_dr: Position = (
            self.position_true
        )  # Dead reckoning position, we assume that the rover move without noise
        self.position_est: Position = (
            self.position_true
        )  # Estimated position of the rover with PF

        self.position_pred: Position = ( self.position_true)
        
        
        self.max_speed: float = max_speed

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

    def pf_localization(self, speed, rotation,):
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

    

    def ekf_localization(self, speed, rotation,xEst,xTrue,PEst,xDR):
        u = np.array([[speed],
                     [rotation]])
        DT = 1                  # Pas de simulation
        def observation(xTrue, xd, u):
            # xTrue = motion_model(xTrue, u)
            xTrue = self.move(self.position_true,speed, rotation)
            xTrue = xTrue.to_array(speed,rotation)

            # add noise to gps x-y
            z = observation_model(xTrue)        #remplacer par self.sense
            # z = self.pf_localization(speed,rotation)?
            # + GPS_NOISE @ np.random.randn(2, 1)

            # add noise to input
            ud = u # + INPUT_NOISE @ np.random.randn(2, 1)

            xd = self.move(self.position_true,speed,rotation).to_array(speed,rotation)

            # motion_model(xd, ud)

            return xTrue, z, xd, ud


        def motion_model(x, u):
            F = np.array([[1.0, 0, 0, 0],
                        [0, 1.0, 0, 0],
                        [0, 0, 1.0, 0],
                        [0, 0, 0, 0]])

            B = np.array([[DT * math.cos(x[2, 0]), 0],
                        [DT * math.sin(x[2, 0]), 0],
                        [0.0, DT],
                        [1.0, 0.0]])

            x = F @ x + B @ u

            return x


        def observation_model(x):
            H = np.array([
                [1, 0, 0, 0],
                [0, 1, 0, 0]
            ])

            z = H @ x

            return z


        def jacob_f(x, u):
            """
            Jacobian of Motion Model
            motion model
            x_{t+1} = x_t+v*dt*cos(yaw)
            y_{t+1} = y_t+v*dt*sin(yaw)
            yaw_{t+1} = yaw_t+omega*dt
            v_{t+1} = v{t}
            so
            dx/dyaw = -v*dt*sin(yaw)
            dx/dv = dt*cos(yaw)
            dy/dyaw = v*dt*cos(yaw)
            dy/dv = dt*sin(yaw)
            """
            yaw = x[2, 0]
            v = u[0,0]
            jF = np.array([
                [1.0, 0.0, -DT * v * math.sin(yaw), DT * math.cos(yaw)],
                [0.0, 1.0, DT * v * math.cos(yaw), DT * math.sin(yaw)],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0]])

            return jF


        def jacob_h():
            # Jacobian of Observation Model
            jH = np.array([
                [1, 0, 0, 0],
                [0, 1, 0, 0]
            ])

            return jH


        def ekf_estimation(xEst, PEst, z, u):
            #  Predict
            xPred = motion_model(xEst, u)
            jF = jacob_f(xEst, u)
            PPred = jF @ PEst @ jF.T + Q
            print("xPred : ", xPred)
            #  Update
            jH = jacob_h()
            zPred = observation_model(xPred)
            y = z - zPred
            S = jH @ PPred @ jH.T + R
            K = PPred @ jH.T @ np.linalg.inv(S)
            xEst = xPred + K @ y
            PEst = (np.eye(len(xEst)) - K @ jH) @ PPred
            return xEst, PEst


        def plot_covariance_ellipse(xEst, PEst):  # pragma: no cover
            Pxy = PEst[0:2, 0:2]
            eigval, eigvec = np.linalg.eig(Pxy)

            if eigval[0] >= eigval[1]:
                bigind = 0
                smallind = 1
            else:
                bigind = 1
                smallind = 0
            t = np.arange(0, 2 * math.pi + 0.1, 0.1)
            a = math.sqrt(eigval[bigind])
            b = math.sqrt(eigval[smallind])
            x = [a * math.cos(it) for it in t]
            y = [b * math.sin(it) for it in t]
            angle = math.atan2(eigvec[bigind, 1], eigvec[bigind, 0])
            R = np.array([[math.cos(angle), math.sin(angle)],
                        [-math.sin(angle), math.cos(angle)]])
            fx = R.dot(np.array([[x, y]]))
            px = np.array(fx[0, :] + xEst[0, 0]).flatten()
            py = np.array(fx[1, :] + xEst[1, 0]).flatten()
            plt.plot(px, py, "--r")
        time = 0.0
        # --------
        # State Vector [x y yaw v]'
      # Dead reckoning

        # history
        hxEst = xEst
        hxTrue = xTrue
        hxDR = xTrue
        hz = np.zeros((2, 1))
        xTrue, z, xDR, ud = observation(xTrue, xDR, u)

        xEst, PEst = ekf_estimation(xEst, PEst, z, ud)
        print(xEst)
        x_est_pos = Position(xEst[0][0],xEst[1][0],xEst[2][0])
        print(x_est_pos)
        return x_est_pos




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
                if neighbor[0] < 0 or neighbor[0] > self.planet.size_x() - 1:
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
            # print(f"Moving towards {self.path[0]}")
            delta_x = self.path[0][1] - self.position_est.x
            delta_y = self.path[0][0] - self.position_est.y
            speed = np.sqrt(delta_x**2 + delta_y**2)
            angle_with_horizontal_axis = np.arctan2(delta_y, delta_x) * 180 / np.pi
            rotation = angle_with_horizontal_axis - self.position_est.direction
            return speed, rotation
        return 0, 0

    def time_step(self):
        """Perform a time step (i.e. move rover, sense, locate, etc.)"""

        speed, rotation = self.calc_control()
        speed = min(speed, self.max_speed)  # Speed is capped before adding noise

        self.position_dr = self.move(self.position_dr, speed, rotation)

        # Add noise
        if speed != 0:
            speed_noisy = speed + np.random.normal(0.0, self.noise.speed)
        else:
            speed_noisy = speed

        rotation_noisy = rotation + np.random.normal(0.0, self.noise.rotation)

        # Compute the true position of rover
        self.position_true = self.move(self.position_true, speed_noisy, rotation_noisy)

        # Observation
        self.sense()

        # Localization
        # _init_
        xEst = np.array([[self.position_est.x],[self.position_est.y],[rotation],[speed]])
        xTrue = np.array([[self.position_true.x],[self.position_true.y],[rotation],[speed]])
        PEst = np.eye(4)

        xDR = np.zeros((4, 1))
        self.position_est = self.ekf_localization(speed, rotation,xEst,xTrue,PEst,xDR)

        # Check if we reached waypoint
        if (
            self.path
            and (round(self.position_est.y), round(self.position_est.x)) == self.path[0]
        ):
            self.path.pop(0)


def animate(
    i, rover, lists_pos, ax_plot, ax_scatter, landmarks_ray
):  # pylint: disable=too-many-locals, unused-argument
    """Animate function for matplotlib animation"""
    true_list, est_list, dr_list = lists_pos
    true_path, est_path, dr_path = ax_plot
    true_pos, est_pos, dr_pos = ax_scatter

    rover.time_step()

    true_list.append(rover.position_true)
    est_list.append(rover.position_est)
    dr_list.append(rover.position_dr)

    true_path[0].set_data([pos.x for pos in true_list], [pos.y for pos in true_list])
    est_path[0].set_data([pos.x for pos in est_list], [pos.y for pos in est_list])
    dr_path[0].set_data([pos.x for pos in dr_list], [pos.y for pos in dr_list])
    true_pos.set_offsets([rover.position_true.x, rover.position_true.y])
    est_pos.set_offsets([rover.position_est.x, rover.position_est.y])
    dr_pos.set_offsets([rover.position_dr.x, rover.position_dr.y])

    for k, landmark in enumerate(rover.planet.landmarks):
        landmarks_ray[k][0].set_data(
            [landmark.x, rover.position_true.x], [landmark.y, rover.position_true.y]
        )

    return true_path, est_path, dr_path, true_pos, est_pos, dr_pos, landmarks_ray


def main():  # pylint: disable=too-many-locals
    """Main function for the simulation"""

    # Initialization
    init_pos = Position(0, 0, 0)
    landmarks = [
        Landmark(25, 150),
        Landmark(50, 80),
        Landmark(50, 20),
        Landmark(120, 50),
        Landmark(125, 115),
        Landmark(160, 80),
        Landmark(190, 185),
    ]

    # Load maze
    img_path = "images/map_200.png"
    maze = cv2.bitwise_not(cv2.imread(img_path, 0)) / 255.0
    maze[maze != 0] = 1
    planet = Planet(maze, landmarks)
    noise = Noise(0.5, 5.0, 0.5)
    final_pos = (planet.size_x() - 1, planet.size_y() - 1)
    rover = Rover(init_pos, final_pos, planet, noise)

    true_pos_list = [rover.position_true]
    est_pos_list = [rover.position_est]
    dr_pos_list = [rover.position_dr]

    # Simulation
    fig = plt.figure()
    ax = fig.add_subplot(
        111, xlim=(0, planet.size_x() - 1), ylim=(0, planet.size_y() - 1)
    )
    ax.imshow(maze, cmap="gray_r")

    # Trajectories
    true_path = ax.plot([], [], c="C0", alpha=0.6, label="True trajectory", zorder=1)
    est_path = ax.plot(
        [], [], c="C1", alpha=0.6, label="Estimated trajectory", zorder=1
    )
    dr_path = ax.plot(
        [], [], c="C2", alpha=0.6, label="Dead_reckoning trajectory", zorder=1
    )

    # Actual position
    true_pos = ax.scatter([], [], c="C0", marker="o", s=10, zorder=2)
    est_pos = ax.scatter([], [], c="C1", marker="o", s=10, zorder=2)
    dr_pos = ax.scatter([], [], c="C2", marker="o", s=10, zorder=2)

    # Landmarks
    ax.scatter(
        [l.x for l in rover.planet.landmarks],
        [l.y for l in rover.planet.landmarks],
        c="r",
        marker="x",
    )
    landmarks_ray = [
        ax.plot([], [], c="r", alpha=0.5, zorder=1) for _ in rover.planet.landmarks
    ]

    # Results
    ax.legend()
    anim = animation.FuncAnimation(  # pylint: disable=unused-variable
        fig,
        animate,
        frames=3 * len(rover.path),
        interval=20,
        fargs=(
            rover,
            [true_pos_list, est_pos_list, dr_pos_list],
            [true_path, est_path, dr_path],
            [true_pos, est_pos, dr_pos],
            landmarks_ray,
        ),
    )

    # Save the animation (it can take a while)
    # anim.save("path_planning.mp4", fps=30, dpi=300)

    plt.show()


if __name__ == "__main__":
    main()