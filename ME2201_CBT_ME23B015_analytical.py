# %%
import sympy as sp
import numpy as np
import matplotlib as mpl
import math
import matplotlib.pyplot as plt
from numpy import pi


# %%
t = sp.symbols("t")
PRECISION = 300


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

    def min_width(self, motion, t_range=None):
        max_y_dot = 0
        min_y_dot = 0
        y_dot = sp.diff(motion.y, t)

        if t_range is None:
            t_range = np.arange(0, 2 * np.pi, 0.01)
        for t_val in t_range:
            max_y_dot = max(max_y_dot, y_dot.subs("t", t_val))
            min_y_dot = min(min_y_dot, y_dot.subs("t", t_val))

        return (max_y_dot + abs(min_y_dot)).evalf()

    def radius_of_curvature(self, motion, cam):
        y_dot = sp.diff(motion.y, t)
        y_dot_dot = sp.diff(y_dot, t)
        return cam.R + motion.y + y_dot_dot**2


def find_minima_by_iter(f, t, t_range):
    minima = float("inf")
    minima_t = None
    for t_val in t_range:
        f_val = f.subs(t, t_val)
        if f_val < minima:
            minima = f_val
            minima_t = t_val

    return minima_t, minima


def find_maxima_by_iter(f, t, t_range):
    maxima = -float("inf")
    maxima_t = None
    for t_val in t_range:
        f_val = f.subs(t, t_val)
        if f_val > maxima:
            maxima = f_val
            maxima_t = t_val

    return maxima_t, maxima


theta_ri = 1.7
alpha = 3.6
L = 50
w = 10  # angular velocity in rad/sec
motion = Motion(
    ("3-4-5", Range(0, theta_ri), Range(0, L)),
    (L, Range(theta_ri, 2), Range(L, L)),
    ("3-4-5", Range(2.0, alpha), Range(L, 0)),
    (0, Range(alpha, 2 * pi), Range(0, 0)),
)

# Lift of follower in mm at theta = 1 radian
print(f"2(a) Lift of follower at theta = 1 rad: {motion(1)} mm")

# Lift of the follower in mm at theta = 2.5 radians
print(f"2(b) Lift of follower at theta = 2.5 rad: {motion(2.5)} mm")

vel = sp.diff(motion.y, t) * w
# Velocity of the follower in mm/sec at theta = 1 radian
print(f"2(c) Velocity of follower at theta = 1 rad: {vel.subs(t, 1)} mm/sec")

# Velocity of the follower in mm/sec theta = 2.5 radians
print(f"2(d) Velocity of follower at theta = 2.5 rad: {vel.subs(t, 2.5)} mm/sec")

acc = sp.diff(vel, t) * w
# Acceleration of the follower in mm/sec^2 at theta = 1 radian
print(f"2(e) Acceleration of follower at theta = 1 rad: {acc.subs(t, 1)} mm/sec^2")

# Acceleration of the follower in mm/sec^2 at theta = 2.5 radians
print(f"2(f) Acceleration of follower at theta = 2.5 rad: {acc.subs(t, 2.5)} mm/sec^2")


# Question 3
cam = Cam(radius=50, e=20, direction="CCW")
follower = Roller(radius=10)

# Maximum pressure angle in radians
phi = cam.pressure_angle(motion)
max_phi = find_maxima_by_iter(phi, t, np.arange(0, 2 * pi, 0.01))[1]
print(f"\n3(a) Maximum pressure angle: {max_phi.evalf()} rad")

# Let C be the contact point between the roller and the cam profile at θ = 1 radian.
# Determine the distance (in mm) between the cam center and C
cam_profile = follower.cam_profile(motion, cam)
X_1, Y_1 = cam_profile[0].subs(t, 1).evalf(), cam_profile[1].subs(t, 1).evalf()
dist = math.sqrt(X_1**2 + Y_1**2)
print(f"3(b) Distance between cam center and C at theta = 1 rad: {dist} mm")

# Question 4
# Determine the maximum eccentricity of the driving effort (in mm) during the rise
cam = Cam(radius=40, e=20, direction="CCW")
follower = Flat_Face()
y_dot = sp.diff(motion.y, t)
y_dot_max = find_maxima_by_iter(y_dot, t, np.arange(0, 2 * pi, 0.01))[1]

e_max = 20
epsilon = y_dot_max - e_max
print(f"\n4(a) {epsilon}")

print(f"4(b) {follower.min_width(motion)}")

# Let C be the contact point between the roller and the cam profile at θ = 1 radian.
# Determine the distance (in mm) between the cam center and C
cam_profile = follower.cam_profile(motion, cam)
X_1, Y_1 = cam_profile[0].subs(t, 1).evalf(), cam_profile[1].subs(t, 1).evalf()
dist = math.sqrt(X_1**2 + Y_1**2)
print(f"4(c) Distance between cam center and C at theta = 1 rad: {dist} mm")
