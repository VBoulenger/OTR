""" Main module of the simulation"""

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation

from rover import Position, Rover

matplotlib.use("TkAgg")


def sensing(pos_x, pos_y, maze):
    """WIP"""
    # TODO

    sensor_angles = [0, 45, 90, 135, 180, 225, 270, 315]
    sensor_length = 5

    result = []
    for sensor_orientation in sensor_angles:
        x_sensor = 0
        y_sensor = 0
        while (
            np.sqrt((pos_x - x_sensor) ** 2 + (pos_y - y_sensor) ** 2) < sensor_length
        ):

            result.append(maze[pos_x + x_sensor * np.cos(sensor_orientation)])  # ??


def animate(i, rover, img):  # pylint: disable=unused-argument
    """Animate function for matplotlib.animation"""
    rover.autoguidance_step()
    arr = img.get_array()
    arr[round(rover.position.x), round(rover.position.y), :] = [0.6, 0, 0.3]
    img.set_array(arr)
    return [img]


def main():
    """Main function of the simulation"""

    # Load maze
    img_path = "images/map_200.png"
    img = cv2.imread(img_path, 0)
    img_reverted = cv2.bitwise_not(img)
    new_img = img_reverted / 255.0
    new_img[new_img != 0] = 1

    maze = new_img

    # Define random position of rover
    x_pos = np.random.randint(0, len(maze) - 1)
    y_pos = np.random.randint(0, len(maze) - 1)
    while maze[x_pos, y_pos] == 1:
        x_pos = np.random.randint(0, len(maze) - 1)
        y_pos = np.random.randint(0, len(maze) - 1)

    rover = Rover(Position(0, 0, 0), maze, (x_pos, y_pos))

    # FIXME: Change vizualization to plot (better for error vizualization)
    # Astar position should be seen a waypoint for the rover
    # Create image and random position
    img = np.repeat((1 - maze[:, :, np.newaxis]), 3, axis=2)
    img[x_pos, y_pos, :] = [0.9, 0, 0.1]

    # Display image
    fig = plt.figure()
    img = plt.imshow(img, interpolation="none")
    anim = animation.FuncAnimation(  # pylint: disable=unused-variable
        fig,
        animate,
        frames=200,
        interval=20,
        fargs=[
            rover,
            img,
        ],
    )
    plt.show()


if __name__ == "__main__":
    main()
