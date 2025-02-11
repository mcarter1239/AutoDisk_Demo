import numpy as np

def unit_vector(vector):
    return vector / np.linalg.norm(vector)

def angle_between(v1, v2):
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.degrees(np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0)))

def angle_of(disk):
    angle = np.degrees(np.arctan2(disk[0], disk[1]))
    if angle < 0:
        angle = 360 + angle
    return angle


print(angle_of([-1,1]))
