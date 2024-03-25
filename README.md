# ENPM661-A*-algorithm-for-a-mobile-Robot
This project implements the A* algorithm to find a path for a mobile robot from a start point to a goal point on a given map. The robot has a radius of 5mm and a clearance of 5mm. The code provides visualization of both node exploration and the optimal path generated.

### Features

- A* Algorithm: The code implements the A* search algorithm to find the optimal path from the start point to the goal point.
- Visualization: The code provides a visual representation of the map with explored areas, obstacles, start point, and goal point.
- User Input: Allows users to input start and goal coordinates, clearance, robot radius, and step size.
- Path Planning: The algorithm plans a collision-free path for the robot considering the given map and obstacles.

### Requirements

- Python 3.x
- numpy
- matplotlib
- time
- queue

### Usage

1. Clone the repository:
git clone <repository-url>


2. Run the code:
python A_star_algorithm.py


3. Follow the on-screen instructions to input start and goal coordinates, clearance, robot radius, and step size.

### Sample Map

The provided map is a 2D grid with obstacles represented as filled cells. The start and goal points are marked on the map with different symbols.
