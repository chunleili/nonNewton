import sys, os

sys.path.append(os.getcwd())

from main import *


def visualizer(pts_ti):
    window = ti.ui.Window("SPH", (1024, 1024), show_window=True, vsync=True)

    scene = ti.ui.Scene()
    camera = ti.ui.Camera()
    camera.position(4, 2, 3)
    camera.up(0.0, 1.0, 0.0)
    camera.lookat(-2.3, -0.6, -1.1)
    camera.fov(45)
    scene.set_camera(camera)

    canvas = window.get_canvas()
    radius = 0.002
    movement_speed = 0.02
    background_color = BLACK
    particle_color = WHITE

    box_anchors, box_lines_indices = make_domainbox()

    cnt = 0
    meta.paused = True
    meta.step_num = 0
    while window.running:
        for e in window.get_events(ti.ui.PRESS):
            if e.key == ti.ui.SPACE:
                meta.paused = not meta.paused
                print("paused:", meta.paused)
            if e.key == "f":
                print("Step once, step: ", meta.step_num)
                meta.step_num += 1
        if not meta.paused:
            meta.step_num += 1

        # print(camera.curr_position)
        # print(camera.curr_lookat)
        camera.track_user_inputs(window, movement_speed=movement_speed, hold_key=ti.ui.RMB)
        scene.set_camera(camera)
        scene.point_light((2.0, 2.0, 2.0), color=(1.0, 1.0, 1.0))
        # scene.particles(meta.pd.x, radius=meta.parm.particle_radius, color=WHITE)
        scene.particles(pts_ti, radius=radius, color=WHITE)
        scene.lines(box_anchors, indices=box_lines_indices, color=(0.99, 0.68, 0.28), width=1.0)
        canvas.scene(scene)
        cnt += 1
        window.show()


def test_transform():
    cfg = get_solid_cfg()[0]
    mesh_path = sph_root_path + cfg["geometryFile"]
    pts = read_ply_particles(mesh_path)
    pts = transform(pts, cfg)

    pts_ti = ti.Vector.field(3, dtype=ti.f32, shape=pts.shape[0])
    pts_ti.from_numpy(pts)
    visualizer(pts_ti)


def test_fill_tank():
    meta.parm = Parameter()
    pos = fill_tank(1.0)
    pos_ti = ti.Vector.field(3, dtype=ti.f32, shape=pos.shape[0])
    pos_ti.from_numpy(pos)
    visualizer(pos_ti)


def test_neighborhoodSearchSparse():
    meta.parm = Parameter()
    pts = read_ply_particles(sph_root_path + "/data/models/test_neighbor.ply")
    meta.pd = ParticleData(pts.shape[0])
    meta.pd.x.from_numpy(pts)
    meta.particle_max_num = pts.shape[0]
    meta.ns = NeighborhoodSearchSparse()
    meta.ns.run_search()
    np.savetxt("pts.txt", pts)
    np.savetxt("neighbors.txt", meta.ns.neighbors.to_numpy(), fmt="%d")
    np.savetxt("num_neighbors.txt", meta.ns.num_neighbors.to_numpy(), fmt="%d")
    np.savetxt("grid_particles_num.txt", meta.ns.grid_particles_num.to_numpy().reshape(-1, 1), fmt="%d")
    p_in_g = meta.ns.particles_in_grid.to_numpy().reshape(-1, 1)
    p_in_g = p_in_g[p_in_g != -1]
    np.savetxt("particles_in_grid.txt", p_in_g, fmt="%d")


if __name__ == "__main__":
    # test_transform()
    # test_fill_tank()
    test_neighborhoodSearchSparse()
