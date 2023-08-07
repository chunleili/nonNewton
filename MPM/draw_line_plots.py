import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange


def reassemble_stress_and_compute_norm(Tp_in):
    # The stress is [sxx, syy, szz, sxy, sxz, syz], we transfer it to 3x3 tensor, such as
    # T = [[sxx, sxy, sxz], [sxy, syy, syz], [sxz, syz, szz]]
    Tp_out = np.zeros((Tp_in.shape[0], 3, 3))
    Tp_norm = np.zeros(Tp_in.shape[0])
    for i in range(Tp_in.shape[0]):
        Tp_out[i, 0, 0] = Tp_in[i, 0]
        Tp_out[i, 1, 1] = Tp_in[i, 1]
        Tp_out[i, 2, 2] = Tp_in[i, 2]
        Tp_out[i, 0, 1] = Tp_in[i, 3]
        Tp_out[i, 1, 0] = Tp_in[i, 3]
        Tp_out[i, 0, 2] = Tp_in[i, 4]
        Tp_out[i, 2, 0] = Tp_in[i, 4]
        Tp_out[i, 1, 2] = Tp_in[i, 5]
        Tp_out[i, 2, 1] = Tp_in[i, 5]
        Tp_norm[i] = np.linalg.norm(Tp_out[i])
    Tp_out = Tp_out.reshape(Tp_out.shape[0], 9)
    return Tp_out, Tp_norm


def load(filename: str):
    if filename.endswith(".npy"):
        return np.load(filename)
    else:
        return np.loadtxt(filename)


def proccess_stress():
    num_frames = 200
    Tps = []
    for frame in trange(num_frames):
        Tp = load(f"results/Tp_{frame}.txt")
        Tp_full, Tp_norm = reassemble_stress_and_compute_norm(Tp)
        Tp_avg_norm = np.mean(Tp_norm)
        Tps.append(Tp_avg_norm)

    plt.figure(figsize=(10, 6))
    plt.plot(Tps, label="stress avg norm", marker="o", markersize=1, color="blue")
    plt.ylabel("stress norm")
    plt.xlabel("step num")
    plt.title("MPM")
    plt.legend()
    plt.savefig("results/stress.png")
    plt.show()
    np.savetxt("results/stress.txt", Tps)


def proccess_velocity():
    num_frames = 200
    vps = []
    for frame in trange(num_frames):
        vp = load(f"results/vp_{frame}.txt")
        vp_norm = np.linalg.norm(vp)
        vps.append(vp_norm)

    plt.figure(figsize=(10, 6))
    plt.plot(vps, label="vp norm", marker=".", markersize=1, color="red")
    plt.xlabel("step num")
    plt.xlabel("vp norm")
    plt.legend()
    plt.title("MPM")
    plt.savefig("results/velocity.png")
    plt.show()
    np.savetxt("results/velocity.txt", vps)


if __name__ == "__main__":
    proccess_velocity()
    proccess_stress()
