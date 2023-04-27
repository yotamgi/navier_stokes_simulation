# importing libraries
import numpy as np

import plot

RHO_TO_P = 10.
DIMS = 2
PIXEL_SIZE = 1
MU = 2.
MAX_POWER = 100

WINDOW_NAME = 'Navier Stokes'
WINDOW_SIZE = (1200, 1200)

rho = np.ones((300, 300), dtype=float)
u = np.zeros((DIMS, *rho.shape), dtype=float)
F = np.zeros_like(u)
constraints = np.zeros_like(u)
mul_constraints = np.ones_like(u[0,...])
#rho[130:180, :] = 0.9

#u[1, 130:180, 130:180] = 1.

#F[1, 40:60, 30:70] = -1.
#F[1, 40:60, 30:70] = -1.
#F[0, 110:190, 110:190] = -1.
#F[1, 110:190, 110:190] = -1.
F[1, 130:170, 110-60:190-60] = 0.1

# Set the line constraints
#rho[129, 110:190] = 0.5
#mul_constraints[129, 110:190] = 0.0
#constraints[:, 128, 110:190] = np.array([1, 0])[:, None]
#constraints[:, 130, 110:190] = np.array([-1, 0])[:, None]

#constraints[:, 127:130, 110:190] = np.array([1, 0])[:, None, None]
#constraints[:, 170:173, 110:190] = np.array([1, 0])[:, None, None]

# Set the border constraints
mul_constraints[:, 0] = 0
#rho[:, 0] = 0.5
constraints[:, :, 1] = np.array([0, -1])[:, None]

mul_constraints[:, -1] = 0
#rho[:, -1] = 0.5
constraints[:, :, -2] = np.array([0, 1])[:, None]

mul_constraints[0, :] = 0
#rho[0, :] = 0.5
constraints[:, 1, :] = np.array([-1, 0])[:, None]

mul_constraints[-1, :] = 0
#rho[-1, :] = 0.5
constraints[:, -2, :] = np.array([1, 0])[:, None]

mul_constraints[-2, -2] = 0
mul_constraints[ 1, -2] = 0
mul_constraints[-2,  1] = 0
mul_constraints[ 1,  1] = 0

#for x in range(40):
#    F[1, 130+x, 110:190] = (
#            np.exp(-np.arange(-40, 40) ** 2 / 20**2) *
#            np.exp(-(x-20) ** 2 / 10**2)
#    ) / 2


def shift_array(array: np.ndarray, shift: int, axis: int) -> np.ndarray:
    """Shift a numpy array by a constant amount on a specified axis."""
    return np.roll(array, shift, axis=axis)

def d(x, i):
    #return (x - shift_array(x, -1, i)) / PIXEL_SIZE
    #return (shift_array(x, 1, i) - shift_array(x, -1, i)) / (2*PIXEL_SIZE)
    #shifted_right = shift_array(x, 1, i) 
    #shifted_left = shift_array(x, -1, i) 
    return (shift_array(x, -1, i) - shift_array(x, 1, i)) / (2*PIXEL_SIZE)

def d_left(x, i):
    return (shift_array(x, -1, i) - x) / PIXEL_SIZE
def d_right(x, i):
    return (x - shift_array(x, 1, i)) / PIXEL_SIZE
def apply_continuity(x, u, i):
    u_positive = np.maximum(u, 0)
    u_negative = np.minimum(u, 0)
    return d_left(u_negative*x, i) + d_right(u_positive*x, i)

def balanced_d(x, u, i):
    return np.where(u > 0, d_left(x, i), d_right(x, i))

momentum = u * rho
  
video_window = plot.VideoWindow(WINDOW_NAME, WINDOW_SIZE)

frame_index = 0
while True:
    frame_index += 1
    dt = 0.04

    # Update the force
    power = np.sum(F * u)
    actual_F = MAX_POWER * F / max(power, MAX_POWER)

    # Update u by rho
    p = RHO_TO_P * rho
    for i in range(DIMS):
        left_pressure_force = -np.minimum(d_left(p, i), 0)
        right_pressure_force = -np.maximum(d_right(p, i), 0)
        pressure_force = left_pressure_force + right_pressure_force
        #pressure_force = -balanced_d(p, u[i], i)
        pressure_force = 0.8*pressure_force + -0.2*balanced_d(p, u[i], i)

        momentum[i] += dt * (
                # Continuity:
                - sum([apply_continuity(momentum[i], u[j], j) for j in range(DIMS)])
                # Pressure forces:
                + pressure_force
                # Viscucity forces:
                   + MU * sum([d_right(d_left(u[i], j), j) for j in range(DIMS)])
                   + 4/3 * MU * d_right(sum(d_left(u[j], j) for j in range(DIMS)) , i)
                # External forces:
                + actual_F[i]
            #)
        )
    #u = momentum / rho

    # Update rho by u
    rho += -dt * sum([apply_continuity(rho, u[j], j) for j in range(DIMS)])
    u = momentum / rho

    # apply constraints
    u -= np.maximum(np.einsum('ijk,ijk->jk', u, constraints), 0) * constraints
    u *= mul_constraints

    if frame_index % 10 == 0:
        print(frame_index, ":", np.min(rho))
        #print(frame_index, ":", np.sum(actual_F * u))
        video_window.init_frame()
        video_window.set_image(rho, vmin=0.95, vmax=1.05)
        video_window.plot_vector(u*30, 10)
        if not video_window.show():
            break
