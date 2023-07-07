# from https://github.com/AppledoreM/taichi-surface_reconstruction

import taichi as ti
import numpy as np
import meshio
from srtool import SRTool


if __name__ == "__main__":
    ti.init(arch=ti.cuda, device_memory_GB=4)

    test_point_clouds_np = meshio.read("./SPH/data/models/torus.ply").points
    num_particles = test_point_clouds_np.shape[0]
    test_point_clouds = ti.Vector.field(3, ti.f32, num_particles)
    test_point_clouds.from_numpy(test_point_clouds_np)

    bounding_box = [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]
    voxel_size = 0.01
    particle_radius = 0.02
    sr = SRTool(
        bounding_box,
        voxel_size=voxel_size,
        particle_radius=particle_radius,
        max_num_vertices=20000,
        max_num_indices=600000,
    )
    sr.dual_contouring(isolevel=0, smooth_radius=0.06, num_particles=num_particles, pos=test_point_clouds)

    window = ti.ui.Window("Example for Surface Reconstruction GUI", (768, 768))
    canvas = window.get_canvas()
    scene = ti.ui.Scene()
    camera = ti.ui.Camera()
    camera.position(5, 2, 2)

    while window.running:
        camera.track_user_inputs(window, movement_speed=0.03, hold_key=ti.ui.RMB)
        scene.set_camera(camera)
        scene.ambient_light((0.8, 0.8, 0.8))
        scene.point_light(pos=(0.5, 1.5, 1.5), color=(1, 1, 1))
        scene.mesh(sr.mesh_vertex, sr.mesh_index, vertex_count=sr.num_vertices[None], index_count=sr.num_indices[None])
        canvas.scene(scene)
        window.show()
