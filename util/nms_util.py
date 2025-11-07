import numpy as np
from math import sin, cos


class Box3d:
    def __init__(self, X, Y, Z, L, W, H, yaw, pitch=0, roll=0):
        self.X = X
        self.Y = Y
        self.Z = Z
        self.L = L
        self.W = W
        self.H = H
        self.yaw = yaw
        self.pitch = pitch
        self.roll = roll
        self.corners = self.get_corners()

    def get_corners(self):
        x_corners = self.L / 2 * np.array([1, 1, 1, 1, -1, -1, -1, -1])
        y_corners = self.W / 2 * np.array([1, -1, -1, 1, 1, -1, -1, 1])
        z_corners = self.H / 2 * np.array([1, 1, -1, -1, 1, 1, -1, -1])
        corners = np.vstack((x_corners, y_corners, z_corners))

        rotation_matrix = self.euler_to_rot(self.yaw, self.pitch, self.roll)
        corners = np.dot(rotation_matrix, corners)

        corners[0, :] = corners[0, :] + self.X
        corners[1, :] = corners[1, :] + self.Y
        corners[2, :] = corners[2, :] + self.Z
        return corners

    @staticmethod
    def euler_to_rot(yaw, pitch, roll):
        P = np.array([[cos(pitch), 0, sin(pitch)],
                      [0, 1, 0],
                      [-sin(pitch), 0, cos(pitch)]])
        R = np.array([[1, 0, 0],
                      [0, cos(roll), -sin(roll)],
                      [0, sin(roll), cos(roll)]])
        Y = np.array([[cos(yaw), -sin(yaw), 0],
                      [sin(yaw), cos(yaw), 0],
                      [0, 0, 1]])
        return np.dot(Y, np.dot(P, R))

    def bottom_corners(self):
        return self.corners[:, [2, 3, 7, 6]]
