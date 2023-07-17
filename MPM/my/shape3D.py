import numpy as np
import taichi as ti


def shape3D(x, y, z):
    N1 = (1.0 - x) * (1 - y) * (1 - z) / 8.0
    N2 = (1.0 + x) * (1 - y) * (1 - z) / 8.0
    N3 = (1.0 + x) * (1 + y) * (1 - z) / 8.0
    N4 = (1.0 - x) * (1 + y) * (1 - z) / 8.0
    N5 = (1.0 - x) * (1 - y) * (1 + z) / 8.0
    N6 = (1.0 + x) * (1 - y) * (1 + z) / 8.0
    N7 = (1.0 + x) * (1 + y) * (1 + z) / 8.0
    N8 = (1.0 - x) * (1 + y) * (1 + z) / 8.0
    N = np.array([N1, N2, N3, N4, N5, N6, N7, N8])
    return N


@ti.func
def shape3D_func(x, y, z):
    N1 = (1.0 - x) * (1 - y) * (1 - z) / 8.0
    N2 = (1.0 + x) * (1 - y) * (1 - z) / 8.0
    N3 = (1.0 + x) * (1 + y) * (1 - z) / 8.0
    N4 = (1.0 - x) * (1 + y) * (1 - z) / 8.0
    N5 = (1.0 - x) * (1 - y) * (1 + z) / 8.0
    N6 = (1.0 + x) * (1 - y) * (1 + z) / 8.0
    N7 = (1.0 + x) * (1 + y) * (1 + z) / 8.0
    N8 = (1.0 - x) * (1 + y) * (1 + z) / 8.0
    N = ti.Vector([N1, N2, N3, N4, N5, N6, N7, N8], float)
    return N
