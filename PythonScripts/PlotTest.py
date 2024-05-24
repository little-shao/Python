import numpy as np
import math
import matplotlib
import matplotlib.pyplot as plt
from RobotArm import *

joints = np.array([
    [1, 1, 0, 1],
    [2, 2, 0, 1],
    [3, 3, 0, 1],
    [4, 4, 0, 1]
], dtype=np.float32).T
fig, ax = plt.subplots(figsize=(5,5))
fig.subplots_adjust(left=0, bottom=0, right=1, top=1)
# targetPt, = ax.plot(0.5, 0.5, marker="o", c="r")
endEff, = ax.plot([], [], marker="o", markerfacecolor="w", c="g", lw=2)
armLine, = ax.plot([], [], marker="o", c="g", lw=2)
armLine.set_data(joints[0, :-1], joints[1, :-1])
endEff.set_data(joints[0, -2::], joints[1, -2::])

ax.add_artist(armLine)
ax.add_artist(endEff)

plt.show()
