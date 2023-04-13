# importing libraries
import numpy as np

import plot

RHO_TO_P = 10.
DIMS = 2
PIXEL_SIZE = 1
MU = 4.0

WINDOW_NAME = 'Navier Stokes'

rho = np.ones((300, 300), dtype=float)
#rho[130:180, :] = 0.9

u = np.zeros((DIMS, *rho.shape), dtype=float)
#u[1, 130:180, 130:180] = 1.

F = np.zeros_like(u)
#F[1, 40:60, 30:70] = -1.
#F[1, 40:60, 30:70] = -1.
#F[0, 110:190, 110:190] = -1.
#F[1, 110:190, 110:190] = -1.
F[1, 110:190, 130:170] = 0.1

# for x in range(80):
#     F[1, 110+x, 110:190] = (
#             np.exp(-np.arange(-40, 40) ** 2 / 20**2) *
#             np.exp(-(x-40) ** 2 / 20**2)
#     )
# 

def shift_array(array: np.ndarray, shift: int, axis: int) -> np.ndarray:
    """Shift a numpy array by a constant amount on a specified axis."""
    return np.roll(array, shift, axis=axis)

def d(x, i):
    #return (x - shift_array(x, -1, i)) / PIXEL_SIZE
    #return (shift_array(x, 1, i) - shift_array(x, -1, i)) / (2*PIXEL_SIZE)
    #shifted_left = shift_array(x, 1, i) 
    #shifted_right = shift_array(x, -1, i) 
    return (shift_array(x, -1, i) - shift_array(x, 1, i)) / (2*PIXEL_SIZE)

def d_right(x, i):
    return (shift_array(x, -1, i) - x)
def d_left(x, i):
    return (x - shift_array(x, 1, i))
def apply_continuity(x, u, i):
    u_positive = np.maximum(u, 0)
    u_negative = np.minimum(u, 0)
    return d_right(u_negative*x, i) + d_left(u_positive*x, i)


momentum = u * rho
  
video_window = plot.VideoWindow(WINDOW_NAME, (1200, 1200))

frame_index = 0
while True:
    frame_index += 1
    dt = 0.01
    
    # Update u by rho
    p = RHO_TO_P * rho
    for i in range(DIMS):
        momentum[i] += dt * (
                # Continuity:
                - sum([apply_continuity(momentum[i], u[j], j) for j in range(DIMS)])
                # Pressure forces:
                - d(p, i)
                # Viscucity forces:
                   + MU * sum([d_left(d_right(u[i], j), j) for j in range(DIMS)])
                   + 4/3 * MU * d_left(sum(d_right(u[j], j) for j in range(DIMS)) , i)
                # External forces:
                + F[i]
            #)
        )
    #u = momentum / rho

    # Update rho by u
    rho += -dt * sum([apply_continuity(rho, u[j], j) for j in range(DIMS)])
    u = momentum / rho

    if frame_index % 10 == 0:
        print(frame_index, ":", np.min(rho))
        video_window.init_frame()
        video_window.set_image(rho, vmin=0.8, vmax=1.2)
        #video_window.set_image(F[1,:,:], vmin=0, vmax=1)
        #video_window.plot_vector(u*30, 10)
        if not video_window.show():
            break
