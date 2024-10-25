import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))
from analytical import plt, pi, Range, Motion, Cam, Flat_Face

fig, ax = plt.subplots()
ax.set_aspect("equal")

steps = [
    (0, Range(0, pi / 2), Range(0, 0)),
    ("345", Range(pi / 2, pi), Range(0, 25)),
    (25, Range(pi, 3 * pi / 2), Range(25, 25)),
    ("345", Range(3 * pi / 2, 2 * pi), Range(25, 0)),
]
motion = Motion(*steps)
cam = Cam(radius=50, e=15, direction="ACW")
cam.plot_base_circle(ax, label="Base Circle")
follower = Flat_Face()
follower.plot_cam_profile(ax, motion, cam, label="Cam Profile")
print("Minimum Width:", follower.min_width(motion))

plt.legend()
plt.grid()
plt.show()
plt.close()
