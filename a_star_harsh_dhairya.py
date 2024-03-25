# -*- coding: utf-8 -*-
"""A_star_harsh_dhairya.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1rlLZ0XuAY-KawViD_hxVxX5ne1AS5s5J
"""

import numpy as np
import matplotlib.pyplot as plt
import time
from queue import Queue
from matplotlib.animation import FuncAnimation
import math

class Vertex:
    def __init__(self, x_coord, y_coord, theta, cost, parent_vertex):
        self.x_coord = x_coord
        self.y_coord = y_coord
        self.theta = theta
        self.cost = cost
        self.parent_vertex = parent_vertex

    def __lt__(self, other):
        return self.cost < other.cost


def move_60up(x,y,theta,step_size, cost):
    theta = theta + 60
    x = x + (step_size*np.cos(np.radians(theta)))
    y = y + (step_size*np.sin(np.radians(theta)))
    #x = round(x*2)/2
    #y = round(y*2)/2
    x = round(x)
    y = round(y)
    cost = 1 + cost
    return x,y,theta,cost

def move_30up(x,y,theta, step_size, cost):
    theta = theta + 30
    x = x + (step_size*np.cos(np.radians(theta)))
    y = y + (step_size*np.sin(np.radians(theta)))
    #x = round(x*2)/2
    #y = round(y*2)/2
    x = round(x)
    y = round(y)
    cost = 1 + cost
    return x,y,theta, cost

def move_0(x,y,theta, step_size, cost):
    theta = theta + 0
    x = x + (step_size*np.cos(np.radians(theta)))
    y = y + (step_size*np.sin(np.radians(theta)))
    #x = round(x*2)/2
    #y = round(y*2)/2
    x = round(x)
    y = round(y)
    cost = 1 + cost
    return x,y,theta, cost

def move_30down(x,y,theta, step_size, cost):
    theta = theta - 30
    x = x + (step_size*np.cos(np.radians(theta)))
    y = y + (step_size*np.sin(np.radians(theta)))
    #x = round(x*2)/2
    #y = round(y*2)/2
    x = round(x)
    y = round(y)
    cost = 1 + cost
    return x,y,theta, cost

def move_60down(x,y,theta, step_size, cost):
    theta = theta - 60
    x = x + (step_size*np.cos(np.radians(theta)))
    y = y + (step_size*np.sin(np.radians(theta)))
    #x = round(x*2)/2
    #y = round(y*2)/2
    x = round(x)
    y = round(y)
    cost = 1 + cost
    return x,y,theta,cost

def move_vertex(move, x_coord, y_coord, theta, step_size, cost):
    if move == '60up':
        return move_60up(x_coord, y_coord, theta, step_size, cost)
    elif move == '30up':
        return move_30up(x_coord, y_coord, theta, step_size, cost)
    elif move == '0':
        return move_0(x_coord, y_coord, theta, step_size, cost)
    elif move == '30down':
        return move_30down(x_coord, y_coord, theta, step_size, cost)
    elif move == '60down':
        return move_60down(x_coord, y_coord, theta, step_size, cost)
    else:
        return None

def create_shape_map(width, height, buffer):
    shape_map = np.full((height, width), 0)
    for y_coord in range(0, height):
        for x_coord in range(0, width):
            rect_1_1_temp = (x_coord + buffer) - (100/2)
            rect_1_2_temp = (y_coord + buffer) - (100/2)
            rect_1_3_temp = (x_coord - buffer) - (175/2)
            rect__4_temp = (y_coord - buffer) - (500/2)

            recta_2_1_temp = (x_coord + buffer) - (275/2)
            recta_2_2_temp = (y_coord + buffer) - 0
            recta_2_3_temp = (x_coord - buffer) - (350/2)
            recta_2_4_temp = (y_coord - buffer) - (400/2)

            hexagon_size_6_temp = (y_coord + buffer) +  0.58*(x_coord + buffer) - (475.098/2)
            hexagon_size_5_temp = (y_coord + buffer) - 0.58*(x_coord - buffer) + (275.002/2)
            hexagon_size_4_temp = (x_coord - buffer) - (779.9/2)
            hexagon_size_3_temp = (y_coord - buffer) + 0.58*(x_coord - buffer) - (775.002/2)
            hexagon_size_2_temp = (y_coord - buffer) - 0.58*(x_coord + buffer) - (24.92/2)
            hexagon_size_1_temp = (x_coord + buffer) - (520.1/2)

            trap_a_temp = (x_coord + buffer) - (900/2)
            trap_b_temp = (x_coord + buffer) - (1020/2)
            trap_c_temp = (x_coord - buffer) - (1100/2)
            trap_d_temp = (y_coord + buffer) - (50/2)
            trap_e_temp = (y_coord - buffer) - (125/2)
            trap_f_temp = (y_coord + buffer) - (375/2)
            trap_g_temp = (y_coord - buffer) - (450/2)

            w1_temp = (y_coord) - (buffer)
            w2_temp = (y_coord) - (250-buffer)
            w3_temp = (x_coord) - (buffer)
            w4_temp = (x_coord) - ((600-buffer))

            if((trap_a_temp > 0 and trap_b_temp < 0 and trap_d_temp > 0 and trap_e_temp < 0) or (trap_b_temp > 0 and trap_c_temp < 0 and trap_d_temp > 0 and trap_g_temp < 0) or (trap_f_temp > 0 and trap_g_temp < 0 and trap_a_temp > 0 and trap_b_temp < 0) or (rect_1_1_temp > 0 and rect_1_2_temp > 0 and rect_1_3_temp < 0 and rect__4_temp < 0) or (recta_2_1_temp > 0 and recta_2_3_temp < 0 and recta_2_4_temp < 0 and recta_2_2_temp > 0) or (hexagon_size_6_temp > 0 and hexagon_size_5_temp > 0 and hexagon_size_4_temp < 0 and hexagon_size_3_temp < 0 and hexagon_size_2_temp < 0 and hexagon_size_1_temp > 0) or (w1_temp < 0) or (w2_temp > 0) or (w3_temp < 0) or (w4_temp > 0) ):
                shape_map[y_coord, x_coord] = 1

            w1_temp = (y_coord) - (5/2)
            w2_temp = (y_coord) - (495/2)
            w3_temp = (x_coord) - (5/2)
            w4_temp = (x_coord) - (1195/2)

            rect_1_1 = (x_coord) - (100/2)
            rect_1_2 = (y_coord) - (100/2)
            rect_1_3 = (x_coord) - (175/2)
            rect__4 = (y_coord) - (500/2)

            recta_2_1 = (x_coord) - (275/2)
            recta_2_2 = (y_coord) - 0
            recta_2_4 = (x_coord) - (350/2)
            recta_2_3 = (y_coord) - (400/2)

            hexagon_size_6 = (y_coord) +  0.58*(x_coord) - (475.098/2)
            hexagon_size_5 = (y_coord) - 0.58*(x_coord) + (275.002/2)
            hexagon_size_4 = (x_coord) - (779.9/2)
            hexagon_size_3 = (y_coord) + 0.58*(x_coord) - (775.002/2)
            hexagon_size_2 = (y_coord) - 0.58*(x_coord) - (24.92/2)
            hexagon_size_1 = (x_coord) - (520.1/2)

            trap_a = (x_coord) - (900/2)
            trap_b = (x_coord) - (1020/2)
            trap_c = (x_coord) - (1100/2)
            trap_d = (y_coord) - (50/2)
            trap_e = (y_coord) - (125/2)
            trap_f = (y_coord) - (375/2)
            trap_g = (y_coord) - (450/2)

            if((hexagon_size_6 > 0 and hexagon_size_5 > 0 and hexagon_size_4 < 0 and hexagon_size_3 < 0 and hexagon_size_2 < 0 and hexagon_size_1 > 0) or (rect_1_1 > 0 and rect_1_2 > 0 and rect_1_3 < 0 and rect__4 < 0 ) or (recta_2_1 > 0  and recta_2_3 < 0 and recta_2_4 < 0 and recta_2_2 > 0) or (trap_a > 0 and trap_b < 0 and trap_d > 0 and trap_e < 0) or (trap_b > 0 and trap_c < 0 and trap_d > 0 and trap_g < 0) or (trap_f > 0 and trap_g < 0 and trap_a > 0 and trap_b < 0)):
                shape_map[y_coord, x_coord] = 2

    return shape_map

def is_goal_reached(current, goal):
    if (current.x_coord == goal.x_coord) and (current.y_coord == goal.y_coord):
        return True
    else:
        return False

def is_valid_move(x, y, shape_map):
    size = shape_map.shape
    if(x >= size[1] or x < 0 or y >= size[0] or y < 0) or (shape_map[y][x] == 1) or (shape_map[y][x] == 2):
        return False
    return True

def generate_unique_id(vertex):
    return vertex.x_coord * 10000 + vertex.y_coord

def euclidean_distance(vertex1, vertex2):
    return np.sqrt((vertex1.x_coord - vertex2.x_coord)**2 + (vertex1.y_coord - vertex2.y_coord)**2)

def theta_threshold(vertex1, vertex2):
    dx = vertex2.x_coord - vertex1.x_coord
    dy = vertex2.y_coord - vertex1.y_coord
    angle = math.atan2(dy, dx) * 180 / math.pi
    return abs(angle)  # Return absolute value

def heuristic(vertex, goal):
    # Euclidean distance heuristic
    return np.sqrt((vertex.x_coord - goal.x_coord)*2 + (vertex.y_coord - goal.y_coord)*2)

def find_shortest_path(start, goal, shape_map, step_size):
    if start.x_coord == goal.x_coord and start.y_coord == goal.y_coord:
        return None, 1

    goal_vertex = goal
    start_vertex = start
    explored_vertices = set()
    open_queue = Queue()  # Using a simple queue instead of a priority queue
    possible_moves = ['60up', '30up', '0', '30down', '60down']

    start_key = generate_unique_id(start_vertex)
    unexplored_vertices = {start_key: start_vertex}
    open_queue.put(start_vertex)

    while not open_queue.empty():
        current_vertex = open_queue.get()

        if current_vertex.x_coord == goal_vertex.x_coord and current_vertex.y_coord == goal_vertex.y_coord:
            goal_vertex.parent_vertex = current_vertex.parent_vertex
            goal_vertex.cost = current_vertex.cost
            print("Goal Vertex found")
            return [], 1

        current_id = generate_unique_id(current_vertex)

        if current_id in explored_vertices:
            continue

        explored_vertices.add(current_id)

        for move in possible_moves:
            x, y, theta, cost = move_vertex(move, current_vertex.x_coord, current_vertex.y_coord, current_vertex.theta, step_size, current_vertex.cost)
            new_vertex = Vertex(x, y, theta, cost, current_vertex)
            new_vertex_id = generate_unique_id(new_vertex)

            if not is_valid_move(new_vertex.x_coord, new_vertex.y_coord, shape_map) or new_vertex_id in explored_vertices:
                continue

            unexplored_vertices[new_vertex_id] = new_vertex
            open_queue.put(new_vertex)

    return [], 0


def is_duplicate(current_vertex, new_vertex, explored_vertices, euclidean_thresh=0.5, theta_thresh=30):
    for vertex in explored_vertices:
        if isinstance(vertex, Vertex):  # Check if vertex is an instance of Vertex class
            if euclidean_distance(vertex, new_vertex) < euclidean_thresh and abs(theta_threshold(current_vertex, new_vertex) - theta_threshold(current_vertex, vertex)) < theta_thresh:
                return True
    return False