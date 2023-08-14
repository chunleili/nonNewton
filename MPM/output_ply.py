import numpy as np
from tqdm import trange

num_frames = 200
is_binary = True

for frame in trange(num_frames):
    if is_binary:
        xp = np.load(f"results/xp_{frame}.npy")
    else:
        xp = np.loadtxt(f"results/xp_{frame}.txt")

    num_points = xp.shape[0]

    outname = f"results/xp_{frame}.ply"

    if is_binary:
        import meshio

        mesh = meshio.Mesh(xp, [])
        mesh.write(outname)

    else:
        header = f"ply\nformat ascii 1.0\nelement vertex {num_points}\nproperty float x\nproperty float y\nproperty float z\nend_header\n"
        with open(outname, "w") as f:
            f.write(header)
            for i in range(xp.shape[0]):
                f.write(f"{xp[i, 0]} {xp[i, 1]} {xp[i, 2]}\n")
