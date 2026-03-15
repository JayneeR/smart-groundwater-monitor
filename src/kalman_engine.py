import numpy as np

class KalmanFilter:
    def __init__(self, Q=0.01, R=0.1):
        self.Q, self.R = Q, R
        self.x, self.P = 0.0, 1.0

    def update(self, z):
        self.P += self.Q
        K = self.P / (self.P + self.R)
        self.x += K * (z - self.x)
        self.P *= (1 - K)
        return self.x
