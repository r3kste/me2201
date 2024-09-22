import matplotlib.pyplot as plt
import numpy as np
import sympy as sp

w, O4x, O4y, h, anggggle = 10, 3.7, -2, 3, 0.51
theta2vals = np.linspace(0, 2 * np.pi, 100)
theta3vals, theta4vals, theta5vals, distx_vals = [], [], [], []
omegas3vals, omegas4vals, omegas5vals, velsxvals = [], [], [], []
alphas3vals, alphas4vals, alphas5vals, accsxvals = [], [], [], []

def inv(A):
    a, b, c, d = A[0][0], A[0][1], A[1][0], A[1][1]
    return np.array([[d, -b], [-c, a]]) / (a * d - b * c)

def compute_jacobian(func, x, args):
    sym = sp.symbols(f"x0:{len(x)}")
    subs = dict(zip(sym, x))
    jacobian = [[float(sp.diff(f, s).subs(subs)) for s in sym] for f in func(sym, *args)]
    return np.array(jacobian, dtype=float)

def solve(func, x0, args, tol=0.01):
    x = np.asarray(x0, dtype=float)
    for _ in range(69):
        f_val = np.asarray(func(x, *args), dtype=float)
        jacobian = compute_jacobian(func, x, args)
        delta_x = inv(jacobian) @ -f_val
        x += delta_x
        if np.linalg.norm(delta_x, ord=2) < tol:
            return x
    return x

def loop1_displacement(vars, theta2):
    theta3, theta41 = vars
    return (1 * sp.cos(theta2) + 4 * sp.cos(theta3) - 3 * sp.cos(theta41) - O4x,
            1 * sp.sin(theta2) + 4 * sp.sin(theta3) - 3 * sp.sin(theta41) - O4y)

def loop1_velocity(vars, omega2, theta2, theta3, theta41):
    omega3, omega41 = vars
    return (omega2 * -sp.sin(theta2) + 4 * omega3 * -sp.sin(theta3) - 3 * omega41 * -sp.sin(theta41),
            omega2 * sp.cos(theta2) + 4 * omega3 * sp.cos(theta3) - 3 * omega41 * sp.cos(theta41))

def loop1_acceleration(vars, alpha2, omega2, theta2, omega3, theta3, omega41, theta41):
    alpha3, alpha41 = vars
    return (omega2**2 * -sp.cos(theta2) + alpha2 * -sp.sin(theta2) + 4 * (omega3**2 * -sp.cos(theta3) + alpha3 * -sp.sin(theta3)) - 3 * (omega41**2 * -sp.cos(theta41) + alpha41 * -sp.sin(theta41)),
            omega2**2 * -sp.sin(theta2) + alpha2 * sp.cos(theta2) + 4 * (omega3**2 * -sp.sin(theta3) + alpha3 * sp.cos(theta3)) - 3 * (omega41**2 * -sp.sin(theta41) + alpha41 * sp.cos(theta41)))

def loop2_displacement(vars, theta4):
    theta5, distx = vars
    return (3 * sp.cos(theta4) + 3.2 * sp.cos(theta4 + anggggle) - 6.5 * sp.cos(theta5) - (distx - O4x),
            3 * sp.sin(theta4) + 3.2 * sp.sin(theta4 + anggggle) - 6.5 * sp.sin(theta5) - (3 - O4y))

def loop2_velocity(vars, omega4, theta4, theta5):
    omega5, velx = vars
    return (3 * omega4 * -sp.sin(theta4) + 3.2 * omega4 * -sp.sin(theta4 + anggggle) - 6.5 * omega5 * -sp.sin(theta5) - velx,
            3 * omega4 * sp.cos(theta4) + 3.2 * omega4 * sp.cos(theta4 + anggggle) - 6.5 * omega5 * sp.cos(theta5))

def loop2_acceleration(vars, alpha4, omega4, theta4, omega5, theta5):
    alpha5, accx = vars
    return (3 * (omega4**2 * -sp.cos(theta4) + alpha4 * -sp.sin(theta4)) + 3.2 * (omega4**2 * -sp.cos(theta4 + anggggle) + alpha4 * -sp.sin(theta4 + anggggle)) - 6.5 * (omega5**2 * -sp.cos(theta5) + alpha5 * -sp.sin(theta5)) - accx,
            3 * (omega4**2 * -sp.sin(theta4) + alpha4 * sp.cos(theta4)) + 3.2 * (omega4**2 * -sp.sin(theta4 + anggggle) + alpha4 * sp.cos(theta4 + anggggle)) - 6.5 * (omega5**2 * -sp.sin(theta5) + alpha5 * sp.cos(theta5)))

def theta3_4():
    global theta3vals, theta4vals
    theta3vals, theta4vals = zip(*[solve(loop1_displacement, (0, 1), args=(t2,)) for t2 in theta2vals])
    theta3vals, theta4vals = np.array(theta3vals) % (2 * np.pi), np.array(theta4vals) % (2 * np.pi)
    return theta3vals, theta4vals

def theta5_x():
    global theta5vals, distx_vals
    theta5vals, distx_vals = zip(*[solve(loop2_displacement, (0, 0), args=(t4,)) for t4 in theta4vals])
    theta5vals, distx_vals = np.array(theta5vals), np.array(distx_vals)
    return theta5vals, distx_vals

def omegas3_4():
    global omegas3vals, omegas4vals
    omegas3vals, omegas4vals = zip(*[solve(loop1_velocity, (0, 0), args=(w, t2, t3, t4)) for t2, t3, t4 in zip(theta2vals, theta3vals, theta4vals)])
    omegas3vals, omegas4vals = np.array(omegas3vals), np.array(omegas4vals)
    return omegas3vals, omegas4vals

def omega5_v_s():
    global omegas5vals, velsxvals
    omegas5vals, velsxvals = zip(*[solve(loop2_velocity, (0, 0), args=(omega4, theta4, theta5)) for omega4, theta4, theta5 in zip(omegas4vals, theta4vals, theta5vals)])
    omegas5vals, velsxvals = np.array(omegas5vals), np.array(velsxvals)
    return omegas5vals, velsxvals

def alpha3_4s():
    global alphas3vals, alphas4vals
    alphas3vals, alphas4vals = zip(*[solve(loop1_acceleration, (0, 0), args=(0, w, t2, omega3, t3, omega4, t4)) for t2, t3, t4, omega3, omega4 in zip(theta2vals, theta3vals, theta4vals, omegas3vals, omegas4vals)])
    alphas3vals, alphas4vals = np.array(alphas3vals), np.array(alphas4vals)
    return alphas3vals, alphas4vals

def alpha5_ax_s():
    global alphas5vals, accsxvals
    alphas5vals, accsxvals = zip(*[solve(loop2_acceleration, (0, 0), args=(alpha4, omega4, t4, omega5, t5)) for alpha4, omega4, t4, omega5, t5 in zip(alphas4vals, omegas4vals, theta4vals, omegas5vals, theta5vals)])
    alphas5vals, accsxvals = np.array(alphas5vals), np.array(accsxvals)
    return alphas5vals, accsxvals

theta3_4(), theta5_x(), omegas3_4(), omega5_v_s(), alpha3_4s(), alpha5_ax_s()

plt.plot(theta2vals, distx_vals, 'b-', label='distx')
plt.legend()
plt.show()
plt.close()

plt.plot(theta2vals, velsxvals, 'g--', label='vel')
plt.legend()
plt.show()
plt.close()

plt.plot(theta2vals, accsxvals, 'r-.', label='acc')
plt.legend()
plt.show()
plt.close()