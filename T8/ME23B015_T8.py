# %%
import sympy as sp
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from sympy import pi

# %%
t = sp.symbols("t")
PRECISION = 100


class Range:
    def __init__(self, start, end):
        self.start, self.end = start, end

    def length(self):
        return self.end - self.start


class Interval:
    def __init__(self, value, t_range, y_range):
        if value in ("345", "3-4-5"):
            value = self._three_four_five(t_range, y_range)
        elif value in ("cycloid", "cycloidal"):
            value = self._cycloid(t_range, y_range)
        else:
            value = sp.sympify(value)
        self.value = value

    def _three_four_five(self, t_range: Range, y_range: Range):
        t = sp.symbols("t")
        var = (t - t_range.start) / t_range.length()
        expr = 10 * var**3 - 15 * var**4 + 6 * var**5
        return y_range.start + y_range.length() * expr

    def _cycloid(self, t_range: Range, y_range: Range):
        var = (t - t_range.start) / t_range.length()
        expr = var - (sp.sin(2 * pi * var) / (2 * pi))
        return y_range.start + y_range.length() * expr

    def __call__(self, t):
        return self.value.subs("t", t)


class Motion:
    def __init__(self, *args):
        y = []
        for value, t_range, y_range in args:
            interval = Interval(value, t_range, y_range)
            y.append((interval, t_range))
        self.y = sp.Piecewise(
            *[
                (interval.value, sp.And(t_range.start <= t, t < t_range.end))
                for interval, t_range in y
            ]
        )

    def __call__(self, t):
        return self.y.subs("t", t)

    def plot_rise(self, ax, **kwargs):
        x = np.linspace(0, 2 * np.pi, PRECISION)
        y = [self.y.subs("t", t) for t in x]
        ax.plot(x, y, **kwargs)

    def plot_velocity(self, ax, **kwargs):
        x = np.linspace(0, 2 * np.pi, PRECISION)
        y_dot = sp.diff(self.y, t)
        y = [y_dot.subs("t", t) for t in x]
        ax.plot(x, y, **kwargs)

    def plot_acceleration(self, ax, **kwargs):
        x = np.linspace(0, 2 * np.pi, PRECISION)
        y_dot = sp.diff(self.y, t)
        y_dot_dot = sp.diff(y_dot, t)
        y = [y_dot_dot.subs("t", t) for t in x]
        ax.plot(x, y, **kwargs)


class Cam:
    def __init__(self, radius, e, direction):
        self.R = radius
        self.e = e
        if direction in ("ACW", "CCW", "anticlockwise"):
            self.direction = 1
        elif direction in ("CW", "clockwise"):
            self.direction = -1

    def plot_base_circle(self, ax, **kwargs):
        x = np.linspace(-self.R, self.R, PRECISION)
        y = np.sqrt(self.R**2 - x**2)
        x = np.concatenate((x, x[::-1]))
        y = np.concatenate((y, -y[::-1]))
        ax.plot(x, y, **kwargs)

    def pressure_angle(self, motion):
        return sp.atan(
            (sp.diff(motion.y, t) - self.e)
            / (sp.sqrt(self.R**2 - self.e**2) + motion.y)
        )

    def plot_pressure_angle(self, ax, motion, **kwargs):
        phi = self.pressure_angle(motion)
        x_phi, y_phi = [], []
        for t in np.linspace(0, 2 * np.pi, PRECISION):
            x_phi.append(t)
            y_phi.append(phi.subs("t", t))

        x_ticks = kwargs.pop("x_ticks", [0, 0.5 * np.pi, np.pi, 1.5 * np.pi, 2 * np.pi])
        x_ticklabels = kwargs.pop(
            "x_ticklabels",
            ["$0$", r"$\frac{\pi}{2}$", r"$\pi$", r"$\frac{3\pi}{2}$", r"$2\pi$"],
        )
        ax.set_xticks(x_ticks)
        ax.set_xticklabels(x_ticklabels)

        ax.plot(x_phi, y_phi, **kwargs)

    def pitch_circle(self, motion):
        X_p = (self.R + motion.y) * sp.sin(t) + self.e * sp.cos(t)
        Y_p = (self.R + motion.y) * sp.cos(t) - self.e * sp.sin(t)
        return X_p * self.direction, Y_p

    def plot_pitch_circle(self, ax, motion, **kwargs):
        X_p, Y_p = self.pitch_circle(motion)
        x_p, y_p = [], []
        for t in np.linspace(0, 2 * np.pi, PRECISION):
            x_p.append(X_p.subs("t", t))
            y_p.append(Y_p.subs("t", t))
        ax.plot(x_p, y_p, **kwargs)


class Follower:
    pass


class Roller(Follower):
    def __init__(self, radius):
        self.R = radius

    def cam_profile(roller, motion, cam):
        phi = cam.pressure_angle(motion)
        X_r = -roller.R * sp.sin(phi) * sp.cos(t) + (
            cam.R + motion.y - roller.R * sp.cos(phi)
        ) * sp.sin(t)
        Y_r = roller.R * sp.sin(phi) * sp.sin(t) + (
            cam.R + motion.y - roller.R * sp.cos(phi)
        ) * sp.cos(t)
        return X_r * cam.direction, Y_r

    def plot_cam_profile(self, ax, motion, cam, **kwargs):
        X_r, Y_r = self.cam_profile(motion, cam)
        x_r, y_r = [], []
        for t in np.linspace(0, 2 * np.pi, PRECISION):
            x_r.append(X_r.subs("t", t))
            y_r.append(Y_r.subs("t", t))
        ax.plot(x_r, y_r, **kwargs)


class Flat_Face(Follower):
    def __init__(self):
        pass

    def cam_profile(roller, motion, cam):
        X_c = (cam.R + motion.y) * sp.sin(t) + sp.diff(motion.y, t) * sp.cos(t)
        Y_c = (cam.R + motion.y) * sp.cos(t) - sp.diff(motion.y, t) * sp.sin(t)
        return X_c * cam.direction, Y_c

    def plot_cam_profile(self, ax, motion, cam, **kwargs):
        X_c, Y_c = self.cam_profile(motion, cam)
        x_c, y_c = [], []
        for t in np.linspace(0, 2 * np.pi, PRECISION):
            x_c.append(X_c.subs("t", t))
            y_c.append(Y_c.subs("t", t))
        ax.plot(x_c, y_c, **kwargs)

    def min_width(self, motion):
        max_y_dot = 0
        min_y_dot = 0
        y_dot = sp.diff(motion.y, t)
        for t_val in np.linspace(0, 2 * np.pi, PRECISION * 10):
            max_y_dot = max(max_y_dot, y_dot.subs("t", t_val))
            min_y_dot = min(min_y_dot, y_dot.subs("t", t_val))

        return (max_y_dot + abs(min_y_dot)).evalf()

mpl.rcParams["text.usetex"] = True
mpl.rcParams["font.family"] = "serif"
mpl.rcParams["font.size"] = 12

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
