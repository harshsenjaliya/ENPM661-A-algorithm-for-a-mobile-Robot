# ENPM661-A*-algorithm-for-a-mobile-Robot
This project implements the A* algorithm to find a path for a mobile robot from a start point to a goal point on a given map. The robot has a radius of 5mm and a clearance of 5mm. The code provides visualization of both node exploration and the optimal path generated.

# Implementation of A* Algorithm for a Mobile Robot

This repository contains Python code to implement the A* algorithm for finding a path between a start and goal point for a mobile robot. The algorithm takes into account obstacles in the environment and generates an optimal path considering the robot's dimensions.

## Contents

- **main.py**: Python script containing the implementation of the A* algorithm and visualization of the pathfinding process.
- **README.md**: This file providing information about the project.

## Prerequisites

- Python 3.x
- Required libraries: `numpy`, `matplotlib`

## Usage

1. Clone the repository to your local machine:

    ```bash
    git clone https://github.com/<username>/Astar-Mobile-Robot.git
    ```

2. Navigate to the project directory:

    ```bash
    cd Astar-Mobile-Robot
    ```

3. Run the main script:

    ```bash
    python main.py
    ```

4. Follow the prompts to input start and goal coordinates, clearance, robot radius, and step size of the robot. The visualization of the optimal path generation between the start and goal points will be displayed.

## Input

- Start Point Coordinates: (X_s, Y_s, Θ_s)
- Goal Point Coordinates: (X_g, Y_g, Θ_g)
- Clearance and Robot Radius
- Step Size of the Robot

## Output

- Visualization of the exploration process and optimal path generation between the start and goal points.
 
