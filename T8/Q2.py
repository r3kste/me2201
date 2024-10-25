import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))
from analytical import plt, pi, Range, Motion, Cam, Roller

fig, ax = plt.subplots()
ax.set_aspect("equal")

steps = [
    ("cycloidal", Range(0, pi), Range(0, 40)),
    ("cycloidal", Range(pi, 2 * pi), Range(40, 0)),
]
motion = Motion(*steps)

cam = Cam(radius=50, e=0, direction="CW")
cam.plot_pitch_circle(ax, motion, label="Pitch Curve")
follower = Roller(radius=10)
follower.plot_cam_profile(ax, motion, cam, label="Cam Profile")

plt.legend()
plt.grid()
plt.show()
plt.close()
