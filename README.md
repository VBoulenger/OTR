# OTR - Optimized Trajectories for Rovers

[![forthebadge made-with-python](http://ForTheBadge.com/images/badges/made-with-python.svg)](https://www.python.org/)

[![linting: pylint](https://img.shields.io/badge/linting-pylint-yellowgreen)](https://github.com/PyCQA/pylint) [![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black) [![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

A Python implementation of the A\* algorithm coupled with a particle filter for localization.

![](https://github.com/VBoulenger/OTR/blob/main/anim.gif)

## Description

This project is a python simulation for a rover exploring a planet. The world is simulated by a 2D black and white image.

A big assumption used here is the fact that the rover has a prior knowledge of this world (a necessary condition for the A\* algorithm).

Once the rover has planned its path, it will travel along it. Noise is added for every action the rover makes. When it moves, a Gaussian noise is added to its speed and to its rotation. A sensing noise is also added to the measurements that the rover makes on the landmarks.

At every step, the rover estimates its position using the particle filter. Based on this estimated position and the path it needs to follow, it will compute its next control input for the following step

## Installation

A few python packages are necessary to run the code:

- [OpenCV](https://opencv.org/) (cv2)
- [matplotlib](https://matplotlib.org/)
- [numpy](https://numpy.org/)

In order to save the animation, the program also use [FFMPEG](https://ffmpeg.org/).

Once the repo is cloned, and you have installed the dependencies, you can directly run the code with:

> $ python main.py

Some map examples are provided in the _images_ folder, you can also easily create one you like using a software like [GIMP](https://www.gimp.org/).
