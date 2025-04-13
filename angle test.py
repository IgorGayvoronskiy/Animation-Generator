from scipy.spatial.transform import Rotation as R
import numpy as np


direction = np.array([0, 0, 1])
# start = [-11.6534438, 83.61688498, 0.25281439]
# end = [-5.25084212, 41.64218307, 5.08589742]
# # start = [-11.6534438, 0.25281439, 83.61688498]
# # end = [-5.25084212, 5.08589742, 41.64218307]
# direction = np.array(end) - np.array(start)
direction = direction / np.linalg.norm(direction)  # Нормализация
print(direction)

base_direction = np.array([0, 1, 0])


# Кватернион для поворота base_direction в direction
rotation, _ = R.align_vectors([direction], [base_direction])
angles = rotation.as_euler('XYZ', degrees=True)  # Преобразуем в углы Эйлера
print(angles[0], angles[1], angles[2])
print(rotation.apply(base_direction))
print(180 - angles[0], 180 - angles[1], 180 - angles[2])