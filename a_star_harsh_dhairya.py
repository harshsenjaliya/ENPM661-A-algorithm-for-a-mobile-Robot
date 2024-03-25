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


def move_up(x, y, cost):
    return x, y + 1, cost + 1

def move_down(x, y, cost):
    return x, y - 1, cost + 1

def move_left(x, y, cost):
    return x - 1, y, cost + 1

def move_right(x, y, cost):
    return x + 1, y, cost + 1

def move_upright(x, y, cost):
    return x + 1, y + 1, cost + np.sqrt(2)

def move_downright(x, y, cost):
    return x + 1, y - 1, cost + np.sqrt(2)

def move_upleft(x, y, cost):
    return x - 1, y + 1, cost + np.sqrt(2)

def move_downleft(x, y, cost):
    return x - 1, y - 1, cost + np.sqrt(2)

def move_vertex(move, x_coord, y_coord, cost):
    if move == 'Up':
        return move_up(x_coord, y_coord, cost)
    elif move == 'UpRight':
        return move_upright(x_coord, y_coord, cost)
    elif move == 'Right':
        return move_right(x_coord, y_coord, cost)
    elif move == 'DownRight':
        return move_downright(x_coord, y_coord, cost)
    elif move == 'Down':
        return move_down(x_coord, y_coord, cost)
    elif move == 'DownLeft':
        return move_downleft(x_coord, y_coord, cost)
    elif move == 'Left':
        return move_left(x_coord, y_coord, cost)
    elif move == 'UpLeft':
        return move_upleft(x_coord, y_coord, cost)
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

            if((trap_a_temp > 0 and trap_b_temp < 0 and trap_d_temp > 0 and trap_e_temp < 0) or (trap_b_temp > 0 and trap_c_temp < 0 and trap_d_temp > 0 and trap_g_temp < 0) or (trap_f_temp > 0 and trap_g_temp < 0 and trap_a_temp > 0 and trap_b_temp < 0) or (rect_1_1_temp > 0 and rect_1_2_temp > 0 and rect_1_3_temp < 0 and rect__4_temp < 0) or (recta_2_1_temp > 0 and recta_2_3_temp < 0 and recta_2_4_temp < 0 and recta_2_2_temp > 0) or (hexagon_size_6_temp > 0 and hexagon_size_5_temp > 0 and hexagon_size_4_temp < 0 and hexagon_size_3_temp < 0 and hexagon_size_2_temp < 0 and hexagon_size_1_temp > 0)):
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

            if((hexagon_size_6 > 0 and hexagon_size_5 > 0 and hexagon_size_4 < 0 and hexagon_size_3 < 0 and hexagon_size_2 < 0 and hexagon_size_1 > 0) or (rect_1_1 > 0 and rect_1_2 > 0 and rect_1_3 < 0 and rect__4 < 0 ) or (recta_2_1 > 0  and recta_2_3 < 0 and recta_2_4 < 0 and recta_2_2 > 0) or (trap_a > 0 and trap_b < 0 and trap_d > 0 and trap_e < 0) or (trap_b > 0 and trap_c < 0 and trap_d > 0 and trap_g < 0) or (trap_f > 0 and trap_g < 0 and trap_a > 0 and trap_b < 0) or (w1_temp < 0) or (w2_temp > 0) or (w3_temp < 0) or (w4_temp > 0)):
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