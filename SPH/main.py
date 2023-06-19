import os
import argparse
import taichi as ti
import numpy as np
import json
import trimesh
import meshio
import plyfile
from functools import reduce

ti.init(arch=ti.gpu, device_memory_fraction=0.5)

sph_root_path = os.path.dirname(os.path.abspath(__file__))


class Meta:
    ...


meta = Meta()

SOLID = 0
FLUID = 1

# ---------------------------------------------------------------------------- #
#                                read json scene                               #
# ---------------------------------------------------------------------------- #


def filedialog():
    import tkinter as tk
    from tkinter import filedialog

    root = tk.Tk()
    root.filename = filedialog.askopenfilename(initialdir=sph_root_path + "/data/scenes", title="Select a File")
    filename = root.filename
    root.destroy()  # close the window
    print("Open scene file: ", filename)
    return filename


meta.scene_path = filedialog()


class SimConfig:
    def __init__(self, scene_path) -> None:
        self.config = None
        with open(scene_path, "r") as f:
            self.config = json.load(f)
        print(json.dumps(self.config, indent=2))

    def get_cfg(self, name, default=None):
        if name not in self.config["Configuration"]:
            return default
        else:
            return self.config["Configuration"][name]


meta.config = SimConfig(meta.scene_path)


def get_cfg(name, default=None):
    return meta.config.get_cfg(name, default)


# ---------------------------------------------------------------------------- #
#                                      io                                      #
# ---------------------------------------------------------------------------- #
def points_from_volume(mesh, particle_seperation=0.02):
    mesh_vox = mesh.voxelized(pitch=particle_seperation).fill()
    point_cloud = mesh_vox.points
    return point_cloud


def read_ply_particles(geometryFile):
    plydata = plyfile.PlyData.read(geometryFile)
    pts = np.stack([plydata["vertex"]["x"], plydata["vertex"]["y"], plydata["vertex"]["z"]], axis=1)
    return pts


# ---------------------------------------------------------------------------- #
#                            other global functions                            #
# ---------------------------------------------------------------------------- #
def build_solver():
    solver_type = get_cfg("simulationMethod")
    # if solver_type == 0:
    #     return WCSPHSolver()
    if solver_type == 4:
        return DFSPHSolver()
    else:
        raise NotImplementedError(f"Solver type {solver_type} has not been implemented.")


# ---------------------------------------------------------------------------- #
#                                particle system                               #
# ---------------------------------------------------------------------------- #
@ti.data_oriented
class ParticleSystem:
    def __init__(self, GGUI=False):
        self.GGUI = GGUI

        self.domain_start = np.array([0.0, 0.0, 0.0])
        self.domain_start = np.array(get_cfg("domainStart"))

        self.domain_end = np.array([1.0, 1.0, 1.0])
        self.domian_end = np.array(get_cfg("domainEnd"))

        self.domain_size = self.domian_end - self.domain_start

        self.dim = len(self.domain_size)
        assert self.dim > 1
        # Simulation method
        self.simulation_method = get_cfg("simulationMethod")

        # Material
        # SOLID = 0
        # FLUID = 1

        self.particle_radius = get_cfg("particleRadius", 0.01)

        self.particle_diameter = 2 * self.particle_radius
        self.support_radius = self.particle_radius * 4.0  # support radius
        self.m_V0 = 0.8 * self.particle_diameter**self.dim

        self.density0 = get_cfg("density0", 1000.0)  # reference density
        self.g = np.array(get_cfg("gravitation", [0.0, -9.8, 0.0]))
        self.viscosity = 0.01  # viscosity
        self.dt = ti.field(float, shape=())
        self.dt[None] = get_cfg("timeStepSize", 1e-4)

        self.particle_num = ti.field(int, shape=())

        # Grid related properties
        self.grid_size = self.support_radius
        self.grid_num = np.ceil(self.domain_size / self.grid_size).astype(int)
        print("grid size: ", self.grid_num)
        self.padding = self.grid_size

        # # # # All objects id and its particle num
        self.object_collection = dict()
        self.object_id_rigid_body = set()

        # ---------------------------------------------------------------------------- #
        #                                load particles                                #
        # ---------------------------------------------------------------------------- #
        self.fluid_particle_num = 0
        fluid_particle_cfgs = meta.config.config.get("FluidParticles", [])
        self.fluid_particles = []
        for i, cfg_i in enumerate(fluid_particle_cfgs):
            f = read_ply_particles(sph_root_path + cfg_i["geometryFile"])
            self.fluid_particles.append(f)
            self.fluid_particle_num += f.shape[0]
        self.particle_num[None] += self.fluid_particle_num

        self.solid_particle_num = 0
        solid_particle_cfgs = meta.config.config.get("SolidParticles", [])
        self.solid_particles = []
        for i, cfg_i in enumerate(solid_particle_cfgs):
            f = read_ply_particles(sph_root_path + cfg_i["geometryFile"])
            self.solid_particles.append(f)
            self.solid_particle_num += f.shape[0]
        self.particle_num[None] += self.solid_particle_num

        self.particle_max_num = self.particle_num[None]

        # Particle num of each grid
        self.grid_particles_num = ti.field(int, shape=int(self.grid_num[0] * self.grid_num[1] * self.grid_num[2]))
        self.grid_particles_num_temp = ti.field(int, shape=int(self.grid_num[0] * self.grid_num[1] * self.grid_num[2]))

        self.prefix_sum_executor = ti.algorithms.PrefixSumExecutor(self.grid_particles_num.shape[0])

        # Particle related properties
        self.object_id = ti.field(dtype=int, shape=self.particle_max_num)
        self.x = ti.Vector.field(self.dim, dtype=float, shape=self.particle_max_num)
        self.x_0 = ti.Vector.field(self.dim, dtype=float, shape=self.particle_max_num)
        self.v = ti.Vector.field(self.dim, dtype=float, shape=self.particle_max_num)
        self.acceleration = ti.Vector.field(self.dim, dtype=float, shape=self.particle_max_num)
        self.m_V = ti.field(dtype=float, shape=self.particle_max_num)
        self.m = ti.field(dtype=float, shape=self.particle_max_num)
        self.density = ti.field(dtype=float, shape=self.particle_max_num)
        self.pressure = ti.field(dtype=float, shape=self.particle_max_num)
        self.material = ti.field(dtype=int, shape=self.particle_max_num)
        self.color = ti.Vector.field(3, dtype=int, shape=self.particle_max_num)
        self.is_dynamic = ti.field(dtype=int, shape=self.particle_max_num)

        if get_cfg("simulationMethod") == 4:
            self.dfsph_factor = ti.field(dtype=float, shape=self.particle_max_num)
            self.density_adv = ti.field(dtype=float, shape=self.particle_max_num)

        # Buffer for sort
        self.object_id_buffer = ti.field(dtype=int, shape=self.particle_max_num)
        self.x_buffer = ti.Vector.field(self.dim, dtype=float, shape=self.particle_max_num)
        self.x_0_buffer = ti.Vector.field(self.dim, dtype=float, shape=self.particle_max_num)
        self.v_buffer = ti.Vector.field(self.dim, dtype=float, shape=self.particle_max_num)
        self.acceleration_buffer = ti.Vector.field(self.dim, dtype=float, shape=self.particle_max_num)
        self.m_V_buffer = ti.field(dtype=float, shape=self.particle_max_num)
        self.m_buffer = ti.field(dtype=float, shape=self.particle_max_num)
        self.density_buffer = ti.field(dtype=float, shape=self.particle_max_num)
        self.pressure_buffer = ti.field(dtype=float, shape=self.particle_max_num)
        self.material_buffer = ti.field(dtype=int, shape=self.particle_max_num)
        self.color_buffer = ti.Vector.field(3, dtype=int, shape=self.particle_max_num)
        self.is_dynamic_buffer = ti.field(dtype=int, shape=self.particle_max_num)

        if get_cfg("simulationMethod") == 4:
            self.dfsph_factor_buffer = ti.field(dtype=float, shape=self.particle_max_num)
            self.density_adv_buffer = ti.field(dtype=float, shape=self.particle_max_num)

        # Grid id for each particle
        self.grid_ids = ti.field(int, shape=self.particle_max_num)
        self.grid_ids_buffer = ti.field(int, shape=self.particle_max_num)
        self.grid_ids_new = ti.field(int, shape=self.particle_max_num)

        # ========== Initialize particles ==========#
        # initialize fluid particles
        self.all_fluid_list = []
        for i in range(len(self.fluid_particles)):
            f = self.fluid_particles[i]
            self.all_fluid_list.append(f)

        # self.all_fluid_np = np.concatenate(self.all_fluid_list, axis=0)

        # initialize solid particles
        self.all_solid_list = []
        for i in range(len(self.solid_particles)):
            f = self.solid_particles[i]
            self.all_solid_list.append(f)
        self.all_par = np.concatenate(self.all_fluid_list + self.all_solid_list, axis=0)

        self.x.from_numpy(self.all_par)
        self.x_0.from_numpy(self.all_par)
        self.m_V.fill(self.m_V0)
        self.m.fill(self.m_V0 * 1000.0)
        self.density.fill(self.density0)
        self.pressure.fill(0.0)
        self.material.fill(FLUID)
        self.color.fill(0)
        self.is_dynamic.fill(1)

        if self.solid_particle_num > 0:
            self.init_solid_particles()

    @ti.kernel
    def init_solid_particles(self):
        for i in range(self.fluid_particle_num, self.fluid_particle_num + self.solid_particle_num):
            self.is_dynamic[i] = 0
            self.material[i] = SOLID

    @ti.func
    def pos_to_index(self, pos):
        return (pos / self.grid_size).cast(int)

    @ti.func
    def flatten_grid_index(self, grid_index):
        return grid_index[0] * self.grid_num[1] * self.grid_num[2] + grid_index[1] * self.grid_num[2] + grid_index[2]

    @ti.func
    def get_flatten_grid_index(self, pos):
        return self.flatten_grid_index(self.pos_to_index(pos))

    @ti.func
    def is_static_rigid_body(self, p):
        return self.material[p] == SOLID and (not self.is_dynamic[p])

    @ti.func
    def is_dynamic_rigid_body(self, p):
        return self.material[p] == SOLID and self.is_dynamic[p]

    @ti.kernel
    def update_grid_id(self):
        for I in ti.grouped(self.grid_particles_num):
            self.grid_particles_num[I] = 0
        for I in ti.grouped(self.x):
            grid_index = self.get_flatten_grid_index(self.x[I])
            self.grid_ids[I] = grid_index
            ti.atomic_add(self.grid_particles_num[grid_index], 1)
        for I in ti.grouped(self.grid_particles_num):
            self.grid_particles_num_temp[I] = self.grid_particles_num[I]

    @ti.kernel
    def counting_sort(self):
        # FIXME: make it the actual particle num
        for i in range(self.particle_max_num):
            I = self.particle_max_num - 1 - i
            base_offset = 0
            if self.grid_ids[I] - 1 >= 0:
                base_offset = self.grid_particles_num[self.grid_ids[I] - 1]
            self.grid_ids_new[I] = ti.atomic_sub(self.grid_particles_num_temp[self.grid_ids[I]], 1) - 1 + base_offset

        for I in ti.grouped(self.grid_ids):
            new_index = self.grid_ids_new[I]
            self.grid_ids_buffer[new_index] = self.grid_ids[I]
            self.object_id_buffer[new_index] = self.object_id[I]
            self.x_0_buffer[new_index] = self.x_0[I]
            self.x_buffer[new_index] = self.x[I]
            self.v_buffer[new_index] = self.v[I]
            self.acceleration_buffer[new_index] = self.acceleration[I]
            self.m_V_buffer[new_index] = self.m_V[I]
            self.m_buffer[new_index] = self.m[I]
            self.density_buffer[new_index] = self.density[I]
            self.pressure_buffer[new_index] = self.pressure[I]
            self.material_buffer[new_index] = self.material[I]
            self.color_buffer[new_index] = self.color[I]
            self.is_dynamic_buffer[new_index] = self.is_dynamic[I]

            if ti.static(self.simulation_method == 4):
                self.dfsph_factor_buffer[new_index] = self.dfsph_factor[I]
                self.density_adv_buffer[new_index] = self.density_adv[I]

        for I in ti.grouped(self.x):
            self.grid_ids[I] = self.grid_ids_buffer[I]
            self.object_id[I] = self.object_id_buffer[I]
            self.x_0[I] = self.x_0_buffer[I]
            self.x[I] = self.x_buffer[I]
            self.v[I] = self.v_buffer[I]
            self.acceleration[I] = self.acceleration_buffer[I]
            self.m_V[I] = self.m_V_buffer[I]
            self.m[I] = self.m_buffer[I]
            self.density[I] = self.density_buffer[I]
            self.pressure[I] = self.pressure_buffer[I]
            self.material[I] = self.material_buffer[I]
            self.color[I] = self.color_buffer[I]
            self.is_dynamic[I] = self.is_dynamic_buffer[I]

            if ti.static(self.simulation_method == 4):
                self.dfsph_factor[I] = self.dfsph_factor_buffer[I]
                self.density_adv[I] = self.density_adv_buffer[I]

    def initialize_particle_system(self):
        self.update_grid_id()
        self.prefix_sum_executor.run(self.grid_particles_num)
        self.counting_sort()

    @ti.func
    def for_all_neighbors(self, p_i, task: ti.template(), ret: ti.template()):
        center_cell = self.pos_to_index(self.x[p_i])
        for offset in ti.grouped(ti.ndrange(*((-1, 2),) * self.dim)):
            grid_index = self.flatten_grid_index(center_cell + offset)
            for p_j in range(self.grid_particles_num[ti.max(0, grid_index - 1)], self.grid_particles_num[grid_index]):
                if p_i[0] != p_j and (self.x[p_i] - self.x[p_j]).norm() < self.support_radius:
                    task(p_i, p_j, ret)


# ---------------------------------------------------------------------------- #
#                                   SPH Base                                   #
# ---------------------------------------------------------------------------- #


@ti.data_oriented
class SPHBase:
    def __init__(self):
        self.g = meta.ps.g
        self.viscosity = meta.ps.viscosity
        self.density_0 = meta.ps.density0
        self.dt = meta.ps.dt

    @ti.func
    def cubic_kernel(self, r_norm):
        res = ti.cast(0.0, ti.f32)
        h = meta.ps.support_radius
        # value of cubic spline smoothing kernel
        k = 1.0
        if meta.ps.dim == 1:
            k = 4 / 3
        elif meta.ps.dim == 2:
            k = 40 / 7 / np.pi
        elif meta.ps.dim == 3:
            k = 8 / np.pi
        k /= h**meta.ps.dim
        q = r_norm / h
        if q <= 1.0:
            if q <= 0.5:
                q2 = q * q
                q3 = q2 * q
                res = k * (6.0 * q3 - 6.0 * q2 + 1)
            else:
                res = k * 2 * ti.pow(1 - q, 3.0)
        return res

    @ti.func
    def cubic_kernel_derivative(self, r):
        h = meta.ps.support_radius
        # derivative of cubic spline smoothing kernel
        k = 1.0
        if meta.ps.dim == 1:
            k = 4 / 3
        elif meta.ps.dim == 2:
            k = 40 / 7 / np.pi
        elif meta.ps.dim == 3:
            k = 8 / np.pi
        k = 6.0 * k / h**meta.ps.dim
        r_norm = r.norm()
        q = r_norm / h
        res = ti.Vector([0.0 for _ in range(meta.ps.dim)])
        if r_norm > 1e-5 and q <= 1.0:
            grad_q = r / (r_norm * h)
            if q <= 0.5:
                res = k * q * (3.0 * q - 2.0) * grad_q
            else:
                factor = 1.0 - q
                res = k * (-factor * factor) * grad_q
        return res

    @ti.func
    def viscosity_force(self, p_i, p_j, r):
        # Compute the viscosity force contribution
        v_xy = (meta.ps.v[p_i] - meta.ps.v[p_j]).dot(r)
        res = (
            2
            * (meta.ps.dim + 2)
            * self.viscosity
            * (meta.ps.m[p_j] / (meta.ps.density[p_j]))
            * v_xy
            / (r.norm() ** 2 + 0.01 * meta.ps.support_radius**2)
            * self.cubic_kernel_derivative(r)
        )
        return res

    def initialize(self):
        meta.ps.initialize_particle_system()
        for r_obj_id in meta.ps.object_id_rigid_body:
            self.compute_rigid_rest_cm(r_obj_id)
        self.compute_static_boundary_volume()
        self.compute_moving_boundary_volume()

    @ti.kernel
    def compute_rigid_rest_cm(self, object_id: int):
        meta.ps.rigid_rest_cm[object_id] = self.compute_com(object_id)

    @ti.kernel
    def compute_static_boundary_volume(self):
        for p_i in ti.grouped(meta.ps.x):
            if not meta.ps.is_static_rigid_body(p_i):
                continue
            delta = self.cubic_kernel(0.0)
            meta.ps.for_all_neighbors(p_i, self.compute_boundary_volume_task, delta)
            meta.ps.m_V[p_i] = (
                1.0 / delta * 3.0
            )  # TODO: the 3.0 here is a coefficient for missing particles by trail and error... need to figure out how to determine it sophisticatedly

    @ti.func
    def compute_boundary_volume_task(self, p_i, p_j, delta: ti.template()):
        if meta.ps.material[p_j] == SOLID:
            delta += self.cubic_kernel((meta.ps.x[p_i] - meta.ps.x[p_j]).norm())

    @ti.kernel
    def compute_moving_boundary_volume(self):
        for p_i in ti.grouped(meta.ps.x):
            if not meta.ps.is_dynamic_rigid_body(p_i):
                continue
            delta = self.cubic_kernel(0.0)
            meta.ps.for_all_neighbors(p_i, self.compute_boundary_volume_task, delta)
            meta.ps.m_V[p_i] = (
                1.0 / delta * 3.0
            )  # TODO: the 3.0 here is a coefficient for missing particles by trail and error... need to figure out how to determine it sophisticatedly

    def substep(self):
        pass

    @ti.func
    def simulate_collisions(self, p_i, vec):
        # Collision factor, assume roughly (1-c_f)*velocity loss after collision
        c_f = 0.5
        meta.ps.v[p_i] -= (1.0 + c_f) * meta.ps.v[p_i].dot(vec) * vec

    @ti.kernel
    def enforce_boundary_2D(self, particle_type: int):
        for p_i in ti.grouped(meta.ps.x):
            if meta.ps.material[p_i] == particle_type and meta.ps.is_dynamic[p_i]:
                pos = meta.ps.x[p_i]
                collision_normal = ti.Vector([0.0, 0.0])
                if pos[0] > meta.ps.domain_size[0] - meta.ps.padding:
                    collision_normal[0] += 1.0
                    meta.ps.x[p_i][0] = meta.ps.domain_size[0] - meta.ps.padding
                if pos[0] <= meta.ps.padding:
                    collision_normal[0] += -1.0
                    meta.ps.x[p_i][0] = meta.ps.padding

                if pos[1] > meta.ps.domain_size[1] - meta.ps.padding:
                    collision_normal[1] += 1.0
                    meta.ps.x[p_i][1] = meta.ps.domain_size[1] - meta.ps.padding
                if pos[1] <= meta.ps.padding:
                    collision_normal[1] += -1.0
                    meta.ps.x[p_i][1] = meta.ps.padding
                collision_normal_length = collision_normal.norm()
                if collision_normal_length > 1e-6:
                    self.simulate_collisions(p_i, collision_normal / collision_normal_length)

    @ti.kernel
    def enforce_boundary_3D(self, particle_type: int):
        for p_i in ti.grouped(meta.ps.x):
            if meta.ps.material[p_i] == particle_type and meta.ps.is_dynamic[p_i]:
                pos = meta.ps.x[p_i]
                collision_normal = ti.Vector([0.0, 0.0, 0.0])
                if pos[0] > meta.ps.domain_size[0] - meta.ps.padding:
                    collision_normal[0] += 1.0
                    meta.ps.x[p_i][0] = meta.ps.domain_size[0] - meta.ps.padding
                if pos[0] <= meta.ps.padding:
                    collision_normal[0] += -1.0
                    meta.ps.x[p_i][0] = meta.ps.padding

                if pos[1] > meta.ps.domain_size[1] - meta.ps.padding:
                    collision_normal[1] += 1.0
                    meta.ps.x[p_i][1] = meta.ps.domain_size[1] - meta.ps.padding
                if pos[1] <= meta.ps.padding:
                    collision_normal[1] += -1.0
                    meta.ps.x[p_i][1] = meta.ps.padding

                if pos[2] > meta.ps.domain_size[2] - meta.ps.padding:
                    collision_normal[2] += 1.0
                    meta.ps.x[p_i][2] = meta.ps.domain_size[2] - meta.ps.padding
                if pos[2] <= meta.ps.padding:
                    collision_normal[2] += -1.0
                    meta.ps.x[p_i][2] = meta.ps.padding

                collision_normal_length = collision_normal.norm()
                if collision_normal_length > 1e-6:
                    self.simulate_collisions(p_i, collision_normal / collision_normal_length)

    @ti.func
    def compute_com(self, object_id):
        sum_m = 0.0
        cm = ti.Vector([0.0, 0.0, 0.0])
        for p_i in range(meta.ps.particle_num[None]):
            if meta.ps.is_dynamic_rigid_body(p_i) and meta.ps.object_id[p_i] == object_id:
                mass = meta.ps.m_V0 * meta.ps.density[p_i]
                cm += mass * meta.ps.x[p_i]
                sum_m += mass
        cm /= sum_m
        return cm

    @ti.kernel
    def compute_com_kernel(self, object_id: int) -> ti.types.vector(3, float):
        return self.compute_com(object_id)

    @ti.kernel
    def solve_constraints(self, object_id: int) -> ti.types.matrix(3, 3, float):
        # compute center of mass
        cm = self.compute_com(object_id)
        # A
        A = ti.Matrix([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
        for p_i in range(meta.ps.particle_num[None]):
            if meta.ps.is_dynamic_rigid_body(p_i) and meta.ps.object_id[p_i] == object_id:
                q = meta.ps.x_0[p_i] - meta.ps.rigid_rest_cm[object_id]
                p = meta.ps.x[p_i] - cm
                A += meta.ps.m_V0 * meta.ps.density[p_i] * p.outer_product(q)

        R, S = ti.polar_decompose(A)

        if all(abs(R) < 1e-6):
            R = ti.Matrix.identity(ti.f32, 3)

        for p_i in range(meta.ps.particle_num[None]):
            if meta.ps.is_dynamic_rigid_body(p_i) and meta.ps.object_id[p_i] == object_id:
                goal = cm + R @ (meta.ps.x_0[p_i] - meta.ps.rigid_rest_cm[object_id])
                corr = (goal - meta.ps.x[p_i]) * 1.0
                meta.ps.x[p_i] += corr
        return R

    # @ti.kernel
    # def compute_rigid_collision(self):
    #     # FIXME: This is a workaround, rigid collision failure in some cases is expected
    #     for p_i in range(meta.ps.particle_num[None]):
    #         if not meta.ps.is_dynamic_rigid_body(p_i):
    #             continue
    #         cnt = 0
    #         x_delta = ti.Vector([0.0 for i in range(meta.ps.dim)])
    #         for j in range(meta.ps.solid_neighbors_num[p_i]):
    #             p_j = meta.ps.solid_neighbors[p_i, j]

    #             if meta.ps.is_static_rigid_body(p_i):
    #                 cnt += 1
    #                 x_j = meta.ps.x[p_j]
    #                 r = meta.ps.x[p_i] - x_j
    #                 if r.norm() < meta.ps.particle_diameter:
    #                     x_delta += (r.norm() - meta.ps.particle_diameter) * r.normalized()
    #         if cnt > 0:
    #             meta.ps.x[p_i] += 2.0 * x_delta # / cnt

    def solve_rigid_body(self):
        for i in range(1):
            for r_obj_id in meta.ps.object_id_rigid_body:
                if meta.ps.object_collection[r_obj_id]["isDynamic"]:
                    R = self.solve_constraints(r_obj_id)

                    if get_cfg("exportObj"):
                        # For output obj only: update the mesh
                        cm = self.compute_com_kernel(r_obj_id)
                        ret = (
                            R.to_numpy()
                            @ (
                                meta.ps.object_collection[r_obj_id]["restPosition"]
                                - meta.ps.object_collection[r_obj_id]["restCenterOfMass"]
                            ).T
                        )
                        meta.ps.object_collection[r_obj_id]["mesh"].vertices = cm.to_numpy() + ret.T

                    # self.compute_rigid_collision()
                    self.enforce_boundary_3D(SOLID)

    def step(self):
        meta.ps.initialize_particle_system()
        self.compute_moving_boundary_volume()
        self.substep()
        self.solve_rigid_body()
        if meta.ps.dim == 2:
            self.enforce_boundary_2D(FLUID)
        elif meta.ps.dim == 3:
            self.enforce_boundary_3D(FLUID)


# ---------------------------------------------------------------------------- #
#                                     DFSPH                                    #
# ---------------------------------------------------------------------------- #


class DFSPHSolver(SPHBase):
    def __init__(self):
        super().__init__()

        self.surface_tension = 0.01

        self.enable_divergence_solver = True

        self.m_max_iterations_v = 100
        self.m_max_iterations = 100

        self.m_eps = 1e-5

        self.max_error_V = 0.1
        self.max_error = 0.05

    @ti.func
    def compute_densities_task(self, p_i, p_j, ret: ti.template()):
        x_i = meta.ps.x[p_i]
        if meta.ps.material[p_j] == FLUID:
            # Fluid neighbors
            x_j = meta.ps.x[p_j]
            ret += meta.ps.m_V[p_j] * self.cubic_kernel((x_i - x_j).norm())
        elif meta.ps.material[p_j] == SOLID:
            # Boundary neighbors
            ## Akinci2012
            x_j = meta.ps.x[p_j]
            ret += meta.ps.m_V[p_j] * self.cubic_kernel((x_i - x_j).norm())

    @ti.kernel
    def compute_densities(self):
        # for p_i in range(meta.ps.particle_num[None]):
        for p_i in ti.grouped(meta.ps.x):
            if meta.ps.material[p_i] != FLUID:
                continue
            meta.ps.density[p_i] = meta.ps.m_V[p_i] * self.cubic_kernel(0.0)
            den = 0.0
            meta.ps.for_all_neighbors(p_i, self.compute_densities_task, den)
            meta.ps.density[p_i] += den
            meta.ps.density[p_i] *= self.density_0

    @ti.func
    def compute_non_pressure_forces_task(self, p_i, p_j, ret: ti.template()):
        x_i = meta.ps.x[p_i]

        ############## Surface Tension ###############
        if meta.ps.material[p_j] == FLUID:
            # Fluid neighbors
            diameter2 = meta.ps.particle_diameter * meta.ps.particle_diameter
            x_j = meta.ps.x[p_j]
            r = x_i - x_j
            r2 = r.dot(r)
            if r2 > diameter2:
                ret -= self.surface_tension / meta.ps.m[p_i] * meta.ps.m[p_j] * r * self.cubic_kernel(r.norm())
            else:
                ret -= (
                    self.surface_tension
                    / meta.ps.m[p_i]
                    * meta.ps.m[p_j]
                    * r
                    * self.cubic_kernel(ti.Vector([meta.ps.particle_diameter, 0.0, 0.0]).norm())
                )

        ############### Viscosoty Force ###############
        d = 2 * (meta.ps.dim + 2)
        x_j = meta.ps.x[p_j]
        # Compute the viscosity force contribution
        r = x_i - x_j
        v_xy = (meta.ps.v[p_i] - meta.ps.v[p_j]).dot(r)

        if meta.ps.material[p_j] == FLUID:
            f_v = (
                d
                * self.viscosity
                * (meta.ps.m[p_j] / (meta.ps.density[p_j]))
                * v_xy
                / (r.norm() ** 2 + 0.01 * meta.ps.support_radius**2)
                * self.cubic_kernel_derivative(r)
            )
            ret += f_v
        elif meta.ps.material[p_j] == SOLID:
            boundary_viscosity = 0.0
            # Boundary neighbors
            ## Akinci2012
            f_v = (
                d
                * boundary_viscosity
                * (self.density_0 * meta.ps.m_V[p_j] / (meta.ps.density[p_i]))
                * v_xy
                / (r.norm() ** 2 + 0.01 * meta.ps.support_radius**2)
                * self.cubic_kernel_derivative(r)
            )
            ret += f_v
            if meta.ps.is_dynamic_rigid_body(p_j):
                meta.ps.acceleration[p_j] += -f_v * meta.ps.density[p_i] / meta.ps.density[p_j]

    @ti.kernel
    def compute_non_pressure_forces(self):
        for p_i in ti.grouped(meta.ps.x):
            if meta.ps.is_static_rigid_body(p_i):
                meta.ps.acceleration[p_i].fill(0.0)
                continue
            ############## Body force ###############
            # Add body force
            d_v = ti.Vector(self.g)
            meta.ps.acceleration[p_i] = d_v
            if meta.ps.material[p_i] == FLUID:
                meta.ps.for_all_neighbors(p_i, self.compute_non_pressure_forces_task, d_v)
                meta.ps.acceleration[p_i] = d_v

    @ti.kernel
    def advect(self):
        # Update position
        for p_i in ti.grouped(meta.ps.x):
            if meta.ps.is_dynamic[p_i]:
                if meta.ps.is_dynamic_rigid_body(p_i):
                    meta.ps.v[p_i] += self.dt[None] * meta.ps.acceleration[p_i]
                meta.ps.x[p_i] += self.dt[None] * meta.ps.v[p_i]

    @ti.kernel
    def compute_DFSPH_factor(self):
        for p_i in ti.grouped(meta.ps.x):
            if meta.ps.material[p_i] != FLUID:
                continue
            sum_grad_p_k = 0.0
            grad_p_i = ti.Vector([0.0 for _ in range(meta.ps.dim)])

            # `ret` concatenates `grad_p_i` and `sum_grad_p_k`
            ret = ti.Vector([0.0 for _ in range(meta.ps.dim + 1)])

            meta.ps.for_all_neighbors(p_i, self.compute_DFSPH_factor_task, ret)

            sum_grad_p_k = ret[3]
            for i in ti.static(range(3)):
                grad_p_i[i] = ret[i]
            sum_grad_p_k += grad_p_i.norm_sqr()

            # Compute pressure stiffness denominator
            factor = 0.0
            if sum_grad_p_k > 1e-6:
                factor = -1.0 / sum_grad_p_k
            else:
                factor = 0.0
            meta.ps.dfsph_factor[p_i] = factor

    @ti.func
    def compute_DFSPH_factor_task(self, p_i, p_j, ret: ti.template()):
        if meta.ps.material[p_j] == FLUID:
            # Fluid neighbors
            grad_p_j = -meta.ps.m_V[p_j] * self.cubic_kernel_derivative(meta.ps.x[p_i] - meta.ps.x[p_j])
            ret[3] += grad_p_j.norm_sqr()  # sum_grad_p_k
            for i in ti.static(range(3)):  # grad_p_i
                ret[i] -= grad_p_j[i]
        elif meta.ps.material[p_j] == SOLID:
            # Boundary neighbors
            ## Akinci2012
            grad_p_j = -meta.ps.m_V[p_j] * self.cubic_kernel_derivative(meta.ps.x[p_i] - meta.ps.x[p_j])
            for i in ti.static(range(3)):  # grad_p_i
                ret[i] -= grad_p_j[i]

    @ti.kernel
    def compute_density_change(self):
        for p_i in ti.grouped(meta.ps.x):
            if meta.ps.material[p_i] != FLUID:
                continue
            ret = ti.Struct(density_adv=0.0, num_neighbors=0)
            meta.ps.for_all_neighbors(p_i, self.compute_density_change_task, ret)

            # only correct positive divergence
            density_adv = ti.max(ret.density_adv, 0.0)
            num_neighbors = ret.num_neighbors

            # Do not perform divergence solve when paritlce deficiency happens
            if meta.ps.dim == 3:
                if num_neighbors < 20:
                    density_adv = 0.0
            else:
                if num_neighbors < 7:
                    density_adv = 0.0

            meta.ps.density_adv[p_i] = density_adv

    @ti.func
    def compute_density_change_task(self, p_i, p_j, ret: ti.template()):
        v_i = meta.ps.v[p_i]
        v_j = meta.ps.v[p_j]
        if meta.ps.material[p_j] == FLUID:
            # Fluid neighbors
            ret.density_adv += meta.ps.m_V[p_j] * (v_i - v_j).dot(
                self.cubic_kernel_derivative(meta.ps.x[p_i] - meta.ps.x[p_j])
            )
        elif meta.ps.material[p_j] == SOLID:
            # Boundary neighbors
            ## Akinci2012
            ret.density_adv += meta.ps.m_V[p_j] * (v_i - v_j).dot(
                self.cubic_kernel_derivative(meta.ps.x[p_i] - meta.ps.x[p_j])
            )

        # Compute the number of neighbors
        ret.num_neighbors += 1

    @ti.kernel
    def compute_density_adv(self):
        for p_i in ti.grouped(meta.ps.x):
            if meta.ps.material[p_i] != FLUID:
                continue
            delta = 0.0
            meta.ps.for_all_neighbors(p_i, self.compute_density_adv_task, delta)
            density_adv = meta.ps.density[p_i] / self.density_0 + self.dt[None] * delta
            meta.ps.density_adv[p_i] = ti.max(density_adv, 1.0)

    @ti.func
    def compute_density_adv_task(self, p_i, p_j, ret: ti.template()):
        v_i = meta.ps.v[p_i]
        v_j = meta.ps.v[p_j]
        if meta.ps.material[p_j] == FLUID:
            # Fluid neighbors
            ret += meta.ps.m_V[p_j] * (v_i - v_j).dot(self.cubic_kernel_derivative(meta.ps.x[p_i] - meta.ps.x[p_j]))
        elif meta.ps.material[p_j] == SOLID:
            # Boundary neighbors
            ## Akinci2012
            ret += meta.ps.m_V[p_j] * (v_i - v_j).dot(self.cubic_kernel_derivative(meta.ps.x[p_i] - meta.ps.x[p_j]))

    @ti.kernel
    def compute_density_error(self, offset: float) -> float:
        density_error = 0.0
        for I in ti.grouped(meta.ps.x):
            if meta.ps.material[I] == FLUID:
                density_error += self.density_0 * meta.ps.density_adv[I] - offset
        return density_error

    @ti.kernel
    def multiply_time_step(self, field: ti.template(), time_step: float):
        for I in ti.grouped(meta.ps.x):
            if meta.ps.material[I] == FLUID:
                field[I] *= time_step

    def divergence_solve(self):
        # TODO: warm start
        # Compute velocity of density change
        self.compute_density_change()
        inv_dt = 1 / self.dt[None]
        self.multiply_time_step(meta.ps.dfsph_factor, inv_dt)

        m_iterations_v = 0

        # Start solver
        avg_density_err = 0.0

        while m_iterations_v < 1 or m_iterations_v < self.m_max_iterations_v:
            avg_density_err = self.divergence_solver_iteration()
            # Max allowed density fluctuation
            # use max density error divided by time step size
            eta = 1.0 / self.dt[None] * self.max_error_V * 0.01 * self.density_0
            # print("eta ", eta)
            if avg_density_err <= eta:
                break
            m_iterations_v += 1
        print(f"DFSPH - iteration V: {m_iterations_v} Avg density err: {avg_density_err}")

        # Multiply by h, the time step size has to be removed
        # to make the stiffness value independent
        # of the time step size

        # TODO: if warm start
        # also remove for kappa v

        self.multiply_time_step(meta.ps.dfsph_factor, self.dt[None])

    def divergence_solver_iteration(self):
        self.divergence_solver_iteration_kernel()
        self.compute_density_change()
        density_err = self.compute_density_error(0.0)
        return density_err / meta.ps.fluid_particle_num

    @ti.kernel
    def divergence_solver_iteration_kernel(self):
        # Perform Jacobi iteration
        for p_i in ti.grouped(meta.ps.x):
            if meta.ps.material[p_i] != FLUID:
                continue
            # evaluate rhs
            b_i = meta.ps.density_adv[p_i]
            k_i = b_i * meta.ps.dfsph_factor[p_i]
            ret = ti.Struct(dv=ti.Vector([0.0 for _ in range(meta.ps.dim)]), k_i=k_i)
            # TODO: if warm start
            # get_kappa_V += k_i
            meta.ps.for_all_neighbors(p_i, self.divergence_solver_iteration_task, ret)
            meta.ps.v[p_i] += ret.dv

    @ti.func
    def divergence_solver_iteration_task(self, p_i, p_j, ret: ti.template()):
        if meta.ps.material[p_j] == FLUID:
            # Fluid neighbors
            b_j = meta.ps.density_adv[p_j]
            k_j = b_j * meta.ps.dfsph_factor[p_j]
            k_sum = (
                ret.k_i + self.density_0 / self.density_0 * k_j
            )  # TODO: make the neighbor density0 different for multiphase fluid
            if ti.abs(k_sum) > self.m_eps:
                grad_p_j = -meta.ps.m_V[p_j] * self.cubic_kernel_derivative(meta.ps.x[p_i] - meta.ps.x[p_j])
                ret.dv -= self.dt[None] * k_sum * grad_p_j
        elif meta.ps.material[p_j] == SOLID:
            # Boundary neighbors
            ## Akinci2012
            if ti.abs(ret.k_i) > self.m_eps:
                grad_p_j = -meta.ps.m_V[p_j] * self.cubic_kernel_derivative(meta.ps.x[p_i] - meta.ps.x[p_j])
                vel_change = -self.dt[None] * 1.0 * ret.k_i * grad_p_j
                ret.dv += vel_change
                if meta.ps.is_dynamic_rigid_body(p_j):
                    meta.ps.acceleration[p_j] += (
                        -vel_change * (1 / self.dt[None]) * meta.ps.density[p_i] / meta.ps.density[p_j]
                    )

    def pressure_solve(self):
        inv_dt = 1 / self.dt[None]
        inv_dt2 = 1 / (self.dt[None] * self.dt[None])

        # TODO: warm start

        # Compute rho_adv
        self.compute_density_adv()

        self.multiply_time_step(meta.ps.dfsph_factor, inv_dt2)

        m_iterations = 0

        # Start solver
        avg_density_err = 0.0

        while m_iterations < 1 or m_iterations < self.m_max_iterations:
            avg_density_err = self.pressure_solve_iteration()
            # Max allowed density fluctuation
            eta = self.max_error * 0.01 * self.density_0
            if avg_density_err <= eta:
                break
            m_iterations += 1
        print(f"DFSPH - iterations: {m_iterations} Avg density Err: {avg_density_err:.4f}")
        # Multiply by h, the time step size has to be removed
        # to make the stiffness value independent
        # of the time step size

        # TODO: if warm start
        # also remove for kappa v

    def pressure_solve_iteration(self):
        self.pressure_solve_iteration_kernel()
        self.compute_density_adv()
        density_err = self.compute_density_error(self.density_0)
        return density_err / meta.ps.fluid_particle_num

    @ti.kernel
    def pressure_solve_iteration_kernel(self):
        # Compute pressure forces
        for p_i in ti.grouped(meta.ps.x):
            if meta.ps.material[p_i] != FLUID:
                continue
            # Evaluate rhs
            b_i = meta.ps.density_adv[p_i] - 1.0
            k_i = b_i * meta.ps.dfsph_factor[p_i]

            # TODO: if warmstart
            # get kappa V
            meta.ps.for_all_neighbors(p_i, self.pressure_solve_iteration_task, k_i)

    @ti.func
    def pressure_solve_iteration_task(self, p_i, p_j, k_i: ti.template()):
        if meta.ps.material[p_j] == FLUID:
            # Fluid neighbors
            b_j = meta.ps.density_adv[p_j] - 1.0
            k_j = b_j * meta.ps.dfsph_factor[p_j]
            k_sum = (
                k_i + self.density_0 / self.density_0 * k_j
            )  # TODO: make the neighbor density0 different for multiphase fluid
            if ti.abs(k_sum) > self.m_eps:
                grad_p_j = -meta.ps.m_V[p_j] * self.cubic_kernel_derivative(meta.ps.x[p_i] - meta.ps.x[p_j])
                # Directly update velocities instead of storing pressure accelerations
                meta.ps.v[p_i] -= self.dt[None] * k_sum * grad_p_j  # ki, kj already contain inverse density
        elif meta.ps.material[p_j] == SOLID:
            # Boundary neighbors
            ## Akinci2012
            if ti.abs(k_i) > self.m_eps:
                grad_p_j = -meta.ps.m_V[p_j] * self.cubic_kernel_derivative(meta.ps.x[p_i] - meta.ps.x[p_j])

                # Directly update velocities instead of storing pressure accelerations
                vel_change = -self.dt[None] * 1.0 * k_i * grad_p_j  # kj already contains inverse density
                meta.ps.v[p_i] += vel_change
                if meta.ps.is_dynamic_rigid_body(p_j):
                    meta.ps.acceleration[p_j] += (
                        -vel_change * 1.0 / self.dt[None] * meta.ps.density[p_i] / meta.ps.density[p_j]
                    )

    @ti.kernel
    def predict_velocity(self):
        # compute new velocities only considering non-pressure forces
        for p_i in ti.grouped(meta.ps.x):
            if meta.ps.is_dynamic[p_i] and meta.ps.material[p_i] == FLUID:
                meta.ps.v[p_i] += self.dt[None] * meta.ps.acceleration[p_i]

    def substep(self):
        self.compute_densities()
        self.compute_DFSPH_factor()
        if self.enable_divergence_solver:
            self.divergence_solve()
        self.compute_non_pressure_forces()
        self.predict_velocity()
        self.pressure_solve()
        self.advect()


meta.ps = ParticleSystem(GGUI=True)


# ---------------------------------------------------------------------------- #
#                                     main                                     #
# ---------------------------------------------------------------------------- #
def main():
    parser = argparse.ArgumentParser(description="SPH Taichi")
    substeps = get_cfg("numberOfStepsPerRenderUpdate")

    ps = meta.ps
    solver = build_solver()
    solver.initialize()

    window = ti.ui.Window("SPH", (1024, 1024), show_window=True, vsync=False)

    scene = ti.ui.Scene()
    camera = ti.ui.Camera()
    camera.position(5.5, 2.5, 4.0)
    camera.up(0.0, 1.0, 0.0)
    camera.lookat(-1.0, 0.0, 0.0)
    camera.fov(70)
    scene.set_camera(camera)

    canvas = window.get_canvas()
    radius = 0.002
    movement_speed = 0.02
    background_color = (0, 0, 0)  # 0xFFFFFF
    particle_color = (1, 1, 1)

    # Draw the lines for domain
    x_max, y_max, z_max = get_cfg("domainEnd")
    box_anchors = ti.Vector.field(3, dtype=ti.f32, shape=8)
    box_anchors[0] = ti.Vector([0.0, 0.0, 0.0])
    box_anchors[1] = ti.Vector([0.0, y_max, 0.0])
    box_anchors[2] = ti.Vector([x_max, 0.0, 0.0])
    box_anchors[3] = ti.Vector([x_max, y_max, 0.0])
    box_anchors[4] = ti.Vector([0.0, 0.0, z_max])
    box_anchors[5] = ti.Vector([0.0, y_max, z_max])
    box_anchors[6] = ti.Vector([x_max, 0.0, z_max])
    box_anchors[7] = ti.Vector([x_max, y_max, z_max])
    box_lines_indices = ti.field(int, shape=(2 * 12))
    for i, val in enumerate([0, 1, 0, 2, 1, 3, 2, 3, 4, 5, 4, 6, 5, 7, 6, 7, 0, 4, 1, 5, 2, 6, 3, 7]):
        box_lines_indices[i] = val

    cnt = 0
    meta.paused = True
    while window.running:
        for e in window.get_events(ti.ui.PRESS):
            if e.key == ti.ui.SPACE:
                meta.paused = not meta.paused
                print("paused:", meta.paused)
        if not meta.paused:
            solver.step()
        camera.track_user_inputs(window, movement_speed=movement_speed, hold_key=ti.ui.LMB)
        scene.set_camera(camera)
        scene.point_light((2.0, 2.0, 2.0), color=(1.0, 1.0, 1.0))
        scene.particles(ps.x, radius=ps.particle_radius, color=particle_color)
        scene.lines(box_anchors, indices=box_lines_indices, color=(0.99, 0.68, 0.28), width=1.0)
        canvas.scene(scene)
        cnt += 1
        window.show()


if __name__ == "__main__":
    main()
