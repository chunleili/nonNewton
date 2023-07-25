import numpy as np
import matplotlib.pyplot as plt


def proccess_stress():
    num_frames = 200
    Tps = []
    for frame in range(num_frames):
        Tp = np.loadtxt(f"results/Tp_{frame}.txt")
        Tp_norm = np.linalg.norm(Tp)
        Tps.append(Tp_norm)

    plt.figure(figsize=(10, 6))
    plt.plot(Tps, label="stress norm", marker="o", markersize=1, color="blue")
    plt.ylabel("stress norm")
    plt.xlabel("step num")
    plt.title("MPM")
    plt.legend()
    plt.show()


def proccess_velocity():
    num_frames = 200
    vps = []
    for frame in range(num_frames):
        vp = np.loadtxt(f"results/vp_{frame}.txt")
        vp_norm = np.linalg.norm(vp)
        vps.append(vp_norm)

    plt.figure(figsize=(10, 6))
    plt.plot(vps, label="vp norm", marker=".", markersize=1, color="red")
    plt.xlabel("step num")
    plt.xlabel("vp norm")
    plt.legend()
    plt.title("MPM")
    plt.show()


proccess_velocity()
