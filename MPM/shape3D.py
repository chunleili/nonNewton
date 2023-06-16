import numpy as np

def shape3D(x, y, z):
    N1 = (1. - x) * (1 - y) * (1 - z) / 8.
    N2 = (1. + x) * (1 - y) * (1 - z) / 8.
    N3 = (1. + x) * (1 + y) * (1 - z) / 8.
    N4 = (1. - x) * (1 + y) * (1 - z) / 8.
    N5 = (1. - x) * (1 - y) * (1 + z) / 8.
    N6 = (1. + x) * (1 - y) * (1 + z) / 8.
    N7 = (1. + x) * (1 + y) * (1 + z) / 8.
    N8 = (1. - x) * (1 + y) * (1 + z) / 8.
    N = np.array([N1, N2, N3, N4, N5, N6, N7, N8])
    return N
