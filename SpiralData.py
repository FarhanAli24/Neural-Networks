import numpy as np
import matplotlib.pyplot as plt

def generate_spiral_data(points_per_class=100):
    theta = np.linspace(0, 4 * np.pi, points_per_class)
    r = theta**2
    x1 = r * np.sin(theta) + np.random.normal(0, 0.1, points_per_class)
    y1 = r * np.cos(theta) + np.random.normal(0, 0.1, points_per_class)
    x2 = r * np.sin(theta + np.pi) + np.random.normal(0, 0.1, points_per_class)
    y2 = r * np.cos(theta + np.pi) + np.random.normal(0, 0.1, points_per_class)
    return (np.vstack((np.column_stack((x1, y1)), np.column_stack((x2, y2)))),
            np.array([0] * points_per_class + [1] * points_per_class))