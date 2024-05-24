import numpy as np
import math
import torch

torch.set_printoptions(profile="full")
np.set_printoptions(threshold=np.inf)

joints = np.array([
    [1, 1, 0, 1],
    [2, 2, 0, 1],
    [3, 3, 0, 1],
    [4, 4, 0, 1]
], dtype=np.float32).T

# print("长度： %d" % (len(joints[0, :]) -1))
print(joints[:, [-1]])

# for i in range(len(joints[0, :] - 1)):
#     print("data %d" % i)
#     print(joints[:3, [i]])
#     print(joints[1, -2::], joints[2, -2::])
#     print(joints[0, :-1], joints[1, :-1])

