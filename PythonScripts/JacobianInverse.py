import numpy as np
import math
import matplotlib
import matplotlib.pyplot as plt
from RobotArm import *

Arm = RobotArm2D()

Arm.add_revolute_link(length=3, thetaInit=math.radians(10))
Arm.add_revolute_link(length=3, thetaInit=math.radians(15))
Arm.add_revolute_link(length=3, thetaInit=math.radians(20))

Arm.update_joint_coords()

print(Arm.joints)

target = Arm.joints[:, [-1]]
"""
fig:代表整个图像
ax：当前活跃的图像
"""
fig, ax = plt.subplots(figsize=(5, 5))
fig.subplots_adjust(left=0, bottom=0, right=1, top=1)
targetPt, = ax.plot([], [], marker='o', c='r')
endEff, = ax.plot([], [], marker='o', markerfacecolor='w', c='g', lw=2)
armLine, = ax.plot([], [], marker='o', c='g', lw=2)

reach = sum(Arm.lengths)

ax.set_xlim(Arm.xRoot - 1.2 * reach, Arm.xRoot + 1.2 * reach)
ax.set_xlim(Arm.yRoot - 1.2 * reach, Arm.yRoot + 1.2 * reach)

circle = plt.Circle((Arm.xRoot, Arm.yRoot), reach, ls="dashed", fill=False)
ax.add_artist(circle)

"""
绘制joint之间的线
"""


def update_plot():
    armLine.set_data(Arm.joints[0, :-1], Arm.joints[1, :-1])
    endEff.set_data(Arm.joints[0, -2::], Arm.joints[1, -2::])


update_plot()

"""
linalg.pinv:求伪逆
"""


def move_to_target():
    global Arm, target, reach

    distPerUpdate = 0.02 * reach
    if np.linalg.norm(target - Arm.joints[:, [-1]]) > 0.02 * reach:
        targetVector = (target - Arm.joints[:, [-1]])[:3]
        targetUnitVector = targetVector / np.linalg.norm(targetVector)
        deltaR = distPerUpdate * targetUnitVector
        J = Arm.get_jacobian()
        JInv = np.linalg.pinv(J)
        deltaTheta = JInv.dot(deltaR)
        Arm.update_theta(deltaTheta)
        Arm.update_joint_coords()
        update_plot()


mode = 1


def on_button_press(event):
    global target, targetPt
    xClick = event.xdata
    yClick = event.ydata

    if mode == 1 and event.button == 1 and isinstance(xClick, float) and isinstance(yClick, float):
        targetPt.set_data(xClick, yClick)
        target = np.array([[xClick, yClick, 0, 1]]).T


fig.canvas.mpl_connect("button_press_event", on_button_press)

exitFlag = False


def on_key_press(event):
    global exitFlag, mode
    if event.key == "enter":
        exitFlag = True
    elif event.key == "shift":
        mode *= -1


fig.canvas.mpl_connect("key_press_event", on_key_press)

plt.ion()
plt.show()

print("Select plot window and press Shift to toogle mode or press Enter to quit.")

t = 0
while not exitFlag:
    if mode == -1:
        targetX = Arm.xRoot + 1.1 * (math.cos(0.12 * t) * reach) * math.cos(t)
        targetY = Arm.yRoot + 1.1 * (math.cos(0.2 * t) * reach) * math.sin(t)
        targetPt = np.array([[targetX, targetY, 0, 1]]).T
        t += 0.025
    move_to_target()
    fig.canvas.get_tk_widget().update()

