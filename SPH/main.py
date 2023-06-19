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


# ---------------------------------------------------------------------------- #
#                                read json scene                               #
# ---------------------------------------------------------------------------- #

def filedialog():
    import tkinter as tk
    from tkinter import filedialog

    root = tk.Tk()
    root.filename = filedialog.askopenfilename(initialdir=sph_root_path+"/data/scenes", title="Select a File")
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
# ---------------------------------------------------------------------------- #
#                                      io                                      #
# ---------------------------------------------------------------------------- #
def points_from_volume(mesh, particle_seperation=0.02):
    mesh_vox = mesh.voxelized(pitch=particle_seperation).fill()
    point_cloud = mesh_vox.points
    return point_cloud


def read_ply_particles(geometryFile):
    plydata = plyfile.PlyData.read(geometryFile)
    pts = np.stack([plydata['vertex']['x'], plydata['vertex']['y'], plydata['vertex']['z']], axis=1)
    return pts

# ---------------------------------------------------------------------------- #
#                                particle system                               #
# ---------------------------------------------------------------------------- #
@ti.data_oriented
class ParticleSystem:
    def __init__(self, config, GGUI=False):
        self.cfg = config
        self.GGUI = GGUI

        self.domain_start = np.array([0.0, 0.0, 0.0])
        self.domain_start = np.array(self.cfg.get_cfg("domainStart"))

        self.domain_end = np.array([1.0, 1.0, 1.0])
        self.domian_end = np.array(self.cfg.get_cfg("domainEnd"))

        self.domain_size = self.domian_end - self.domain_start

        self.dim = len(self.domain_size)
        assert self.dim > 1
        # Simulation method
        self.simulation_method = self.cfg.get_cfg("simulationMethod")

        # Material
        self.material_solid = 0
        self.material_fluid = 1

        self.particle_radius = 0.01  # particle radius
        self.particle_radius = self.cfg.get_cfg("particleRadius")

        self.particle_diameter = 2 * self.particle_radius
        self.support_radius = self.particle_radius * 4.0  # support radius
        self.m_V0 = 0.8 * self.particle_diameter**self.dim

        self.density0 = 1000.0  # reference density

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
        fluid_particle_cfgs = self.cfg.config.get("FluidParticles", [])
        self.fluid_particles = []
        for i, cfg_i in enumerate(fluid_particle_cfgs):
            f = read_ply_particles(sph_root_path + cfg_i["geometryFile"])
            self.fluid_particles.append(f)
            self.fluid_particle_num += f.shape[0]
        self.particle_num[None] += self.fluid_particle_num

        self.solid_particle_num = 0
        solid_particle_cfgs = self.cfg.config.get("SolidParticles", [])
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

        if self.cfg.get_cfg("simulationMethod") == 4:
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

        if self.cfg.get_cfg("simulationMethod") == 4:
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
        self.all_par = np.concatenate(self.all_fluid_list+self.all_solid_list, axis=0)
    
        self.x.from_numpy(self.all_par)
        self.x_0.from_numpy(self.all_par)
        self.m_V.fill(self.m_V0)
        self.m.fill(self.m_V0 * 1000.0)
        self.density.fill(self.density0)
        self.pressure.fill(0.0)
        self.material.fill(self.material_fluid)
        self.color.fill(0)
        self.is_dynamic.fill(1)

        if self.solid_particle_num > 0:
            self.init_solid_particles()

    @ti.kernel
    def init_solid_particles(self):
        for i in range(self.fluid_particle_num, self.fluid_particle_num+self.solid_particle_num):
            self.is_dynamic[i] = 0
            self.material[i] = self.material_solid


    def build_solver(self):
        solver_type = self.cfg.get_cfg("simulationMethod")
        # if solver_type == 0:
        #     return WCSPHSolver(self)
        if solver_type == 4:
            return DFSPHSolver(self)
        else:
            raise NotImplementedError(f"Solver type {solver_type} has not been implemented.")

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
        return self.material[p] == self.material_solid and (not self.is_dynamic[p])

    @ti.func
    def is_dynamic_rigid_body(self, p):
        return self.material[p] == self.material_solid and self.is_dynamic[p]

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
    def __init__(self, particle_system):
        self.ps = particle_system
        self.g = ti.Vector([0.0, -9.81, 0.0])  # Gravity
        if self.ps.dim == 2:
            self.g = ti.Vector([0.0, -9.81])
        self.g = np.array(self.ps.cfg.get_cfg("gravitation"))

        self.viscosity = 0.01  # viscosity

        self.density_0 = 1000.0  # reference density
        self.density_0 = self.ps.cfg.get_cfg("density0")

        self.dt = ti.field(float, shape=())
        self.dt[None] = 1e-4

    @ti.func
    def cubic_kernel(self, r_norm):
        res = ti.cast(0.0, ti.f32)
        h = self.ps.support_radius
        # value of cubic spline smoothing kernel
        k = 1.0
        if self.ps.dim == 1:
            k = 4 / 3
        elif self.ps.dim == 2:
            k = 40 / 7 / np.pi
        elif self.ps.dim == 3:
            k = 8 / np.pi
        k /= h**self.ps.dim
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
        h = self.ps.support_radius
        # derivative of cubic spline smoothing kernel
        k = 1.0
        if self.ps.dim == 1:
            k = 4 / 3
        elif self.ps.dim == 2:
            k = 40 / 7 / np.pi
        elif self.ps.dim == 3:
            k = 8 / np.pi
        k = 6.0 * k / h**self.ps.dim
        r_norm = r.norm()
        q = r_norm / h
        res = ti.Vector([0.0 for _ in range(self.ps.dim)])
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
        v_xy = (self.ps.v[p_i] - self.ps.v[p_j]).dot(r)
        res = (
            2
            * (self.ps.dim + 2)
            * self.viscosity
            * (self.ps.m[p_j] / (self.ps.density[p_j]))
            * v_xy
            / (r.norm() ** 2 + 0.01 * self.ps.support_radius**2)
            * self.cubic_kernel_derivative(r)
        )
        return res

    def initialize(self):
        self.ps.initialize_particle_system()
        for r_obj_id in self.ps.object_id_rigid_body:
            self.compute_rigid_rest_cm(r_obj_id)
        self.compute_static_boundary_volume()
        self.compute_moving_boundary_volume()

    @ti.kernel
    def compute_rigid_rest_cm(self, object_id: int):
        self.ps.rigid_rest_cm[object_id] = self.compute_com(object_id)

    @ti.kernel
    def compute_static_boundary_volume(self):
        for p_i in ti.grouped(self.ps.x):
            if not self.ps.is_static_rigid_body(p_i):
                continue
            delta = self.cubic_kernel(0.0)
            self.ps.for_all_neighbors(p_i, self.compute_boundary_volume_task, delta)
            self.ps.m_V[p_i] = (
                1.0 / delta * 3.0
            )  # TODO: the 3.0 here is a coefficient for missing particles by trail and error... need to figure out how to determine it sophisticatedly

    @ti.func
    def compute_boundary_volume_task(self, p_i, p_j, delta: ti.template()):
        if self.ps.material[p_j] == self.ps.material_solid:
            delta += self.cubic_kernel((self.ps.x[p_i] - self.ps.x[p_j]).norm())

    @ti.kernel
    def compute_moving_boundary_volume(self):
        for p_i in ti.grouped(self.ps.x):
            if not self.ps.is_dynamic_rigid_body(p_i):
                continue
            delta = self.cubic_kernel(0.0)
            self.ps.for_all_neighbors(p_i, self.compute_boundary_volume_task, delta)
            self.ps.m_V[p_i] = (
                1.0 / delta * 3.0
            )  # TODO: the 3.0 here is a coefficient for missing particles by trail and error... need to figure out how to determine it sophisticatedly

    def substep(self):
        pass

    @ti.func
    def simulate_collisions(self, p_i, vec):
        # Collision factor, assume roughly (1-c_f)*velocity loss after collision
        c_f = 0.5
        self.ps.v[p_i] -= (1.0 + c_f) * self.ps.v[p_i].dot(vec) * vec

    @ti.kernel
    def enforce_boundary_2D(self, particle_type: int):
        for p_i in ti.grouped(self.ps.x):
            if self.ps.material[p_i] == particle_type and self.ps.is_dynamic[p_i]:
                pos = self.ps.x[p_i]
                collision_normal = ti.Vector([0.0, 0.0])
                if pos[0] > self.ps.domain_size[0] - self.ps.padding:
                    collision_normal[0] += 1.0
                    self.ps.x[p_i][0] = self.ps.domain_size[0] - self.ps.padding
                if pos[0] <= self.ps.padding:
                    collision_normal[0] += -1.0
                    self.ps.x[p_i][0] = self.ps.padding

                if pos[1] > self.ps.domain_size[1] - self.ps.padding:
                    collision_normal[1] += 1.0
                    self.ps.x[p_i][1] = self.ps.domain_size[1] - self.ps.padding
                if pos[1] <= self.ps.padding:
                    collision_normal[1] += -1.0
                    self.ps.x[p_i][1] = self.ps.padding
                collision_normal_length = collision_normal.norm()
                if collision_normal_length > 1e-6:
                    self.simulate_collisions(p_i, collision_normal / collision_normal_length)

    @ti.kernel
    def enforce_boundary_3D(self, particle_type: int):
        for p_i in ti.grouped(self.ps.x):
            if self.ps.material[p_i] == particle_type and self.ps.is_dynamic[p_i]:
                pos = self.ps.x[p_i]
                collision_normal = ti.Vector([0.0, 0.0, 0.0])
                if pos[0] > self.ps.domain_size[0] - self.ps.padding:
                    collision_normal[0] += 1.0
                    self.ps.x[p_i][0] = self.ps.domain_size[0] - self.ps.padding
                if pos[0] <= self.ps.padding:
                    collision_normal[0] += -1.0
                    self.ps.x[p_i][0] = self.ps.padding

                if pos[1] > self.ps.domain_size[1] - self.ps.padding:
                    collision_normal[1] += 1.0
                    self.ps.x[p_i][1] = self.ps.domain_size[1] - self.ps.padding
                if pos[1] <= self.ps.padding:
                    collision_normal[1] += -1.0
                    self.ps.x[p_i][1] = self.ps.padding

                if pos[2] > self.ps.domain_size[2] - self.ps.padding:
                    collision_normal[2] += 1.0
                    self.ps.x[p_i][2] = self.ps.domain_size[2] - self.ps.padding
                if pos[2] <= self.ps.padding:
                    collision_normal[2] += -1.0
                    self.ps.x[p_i][2] = self.ps.padding

                collision_normal_length = collision_normal.norm()
                if collision_normal_length > 1e-6:
                    self.simulate_collisions(p_i, collision_normal / collision_normal_length)

    @ti.func
    def compute_com(self, object_id):
        sum_m = 0.0
        cm = ti.Vector([0.0, 0.0, 0.0])
        for p_i in range(self.ps.particle_num[None]):
            if self.ps.is_dynamic_rigid_body(p_i) and self.ps.object_id[p_i] == object_id:
                mass = self.ps.m_V0 * self.ps.density[p_i]
                cm += mass * self.ps.x[p_i]
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
        for p_i in range(self.ps.particle_num[None]):
            if self.ps.is_dynamic_rigid_body(p_i) and self.ps.object_id[p_i] == object_id:
                q = self.ps.x_0[p_i] - self.ps.rigid_rest_cm[object_id]
                p = self.ps.x[p_i] - cm
                A += self.ps.m_V0 * self.ps.density[p_i] * p.outer_product(q)

        R, S = ti.polar_decompose(A)

        if all(abs(R) < 1e-6):
            R = ti.Matrix.identity(ti.f32, 3)

        for p_i in range(self.ps.particle_num[None]):
            if self.ps.is_dynamic_rigid_body(p_i) and self.ps.object_id[p_i] == object_id:
                goal = cm + R @ (self.ps.x_0[p_i] - self.ps.rigid_rest_cm[object_id])
                corr = (goal - self.ps.x[p_i]) * 1.0
                self.ps.x[p_i] += corr
        return R

    # @ti.kernel
    # def compute_rigid_collision(self):
    #     # FIXME: This is a workaround, rigid collision failure in some cases is expected
    #     for p_i in range(self.ps.particle_num[None]):
    #         if not self.ps.is_dynamic_rigid_body(p_i):
    #             continue
    #         cnt = 0
    #         x_delta = ti.Vector([0.0 for i in range(self.ps.dim)])
    #         for j in range(self.ps.solid_neighbors_num[p_i]):
    #             p_j = self.ps.solid_neighbors[p_i, j]

    #             if self.ps.is_static_rigid_body(p_i):
    #                 cnt += 1
    #                 x_j = self.ps.x[p_j]
    #                 r = self.ps.x[p_i] - x_j
    #                 if r.norm() < self.ps.particle_diameter:
    #                     x_delta += (r.norm() - self.ps.particle_diameter) * r.normalized()
    #         if cnt > 0:
    #             self.ps.x[p_i] += 2.0 * x_delta # / cnt

    def solve_rigid_body(self):
        for i in range(1):
            for r_obj_id in self.ps.object_id_rigid_body:
                if self.ps.object_collection[r_obj_id]["isDynamic"]:
                    R = self.solve_constraints(r_obj_id)

                    if self.ps.cfg.get_cfg("exportObj"):
                        # For output obj only: update the mesh
                        cm = self.compute_com_kernel(r_obj_id)
                        ret = (
                            R.to_numpy()
                            @ (
                                self.ps.object_collection[r_obj_id]["restPosition"]
                                - self.ps.object_collection[r_obj_id]["restCenterOfMass"]
                            ).T
                        )
                        self.ps.object_collection[r_obj_id]["mesh"].vertices = cm.to_numpy() + ret.T

                    # self.compute_rigid_collision()
                    self.enforce_boundary_3D(self.ps.material_solid)

    def step(self):
        self.ps.initialize_particle_system()
        self.compute_moving_boundary_volume()
        self.substep()
        self.solve_rigid_body()
        if self.ps.dim == 2:
            self.enforce_boundary_2D(self.ps.material_fluid)
        elif self.ps.dim == 3:
            self.enforce_boundary_3D(self.ps.material_fluid)


# ---------------------------------------------------------------------------- #
#                                     DFSPH                                    #
# ---------------------------------------------------------------------------- #


class DFSPHSolver(SPHBase):
    def __init__(self, particle_system):
        super().__init__(particle_system)

        self.surface_tension = 0.01
        self.dt[None] = self.ps.cfg.get_cfg("timeStepSize")

        self.enable_divergence_solver = True

        self.m_max_iterations_v = 100
        self.m_max_iterations = 100

        self.m_eps = 1e-5

        self.max_error_V = 0.1
        self.max_error = 0.05

    @ti.func
    def compute_densities_task(self, p_i, p_j, ret: ti.template()):
        x_i = self.ps.x[p_i]
        if self.ps.material[p_j] == self.ps.material_fluid:
            # Fluid neighbors
            x_j = self.ps.x[p_j]
            ret += self.ps.m_V[p_j] * self.cubic_kernel((x_i - x_j).norm())
        elif self.ps.material[p_j] == self.ps.material_solid:
            # Boundary neighbors
            ## Akinci2012
            x_j = self.ps.x[p_j]
            ret += self.ps.m_V[p_j] * self.cubic_kernel((x_i - x_j).norm())

    @ti.kernel
    def compute_densities(self):
        # for p_i in range(self.ps.particle_num[None]):
        for p_i in ti.grouped(self.ps.x):
            if self.ps.material[p_i] != self.ps.material_fluid:
                continue
            self.ps.density[p_i] = self.ps.m_V[p_i] * self.cubic_kernel(0.0)
            den = 0.0
            self.ps.for_all_neighbors(p_i, self.compute_densities_task, den)
            self.ps.density[p_i] += den
            self.ps.density[p_i] *= self.density_0

    @ti.func
    def compute_non_pressure_forces_task(self, p_i, p_j, ret: ti.template()):
        x_i = self.ps.x[p_i]

        ############## Surface Tension ###############
        if self.ps.material[p_j] == self.ps.material_fluid:
            # Fluid neighbors
            diameter2 = self.ps.particle_diameter * self.ps.particle_diameter
            x_j = self.ps.x[p_j]
            r = x_i - x_j
            r2 = r.dot(r)
            if r2 > diameter2:
                ret -= self.surface_tension / self.ps.m[p_i] * self.ps.m[p_j] * r * self.cubic_kernel(r.norm())
            else:
                ret -= (
                    self.surface_tension
                    / self.ps.m[p_i]
                    * self.ps.m[p_j]
                    * r
                    * self.cubic_kernel(ti.Vector([self.ps.particle_diameter, 0.0, 0.0]).norm())
                )

        ############### Viscosoty Force ###############
        d = 2 * (self.ps.dim + 2)
        x_j = self.ps.x[p_j]
        # Compute the viscosity force contribution
        r = x_i - x_j
        v_xy = (self.ps.v[p_i] - self.ps.v[p_j]).dot(r)

        if self.ps.material[p_j] == self.ps.material_fluid:
            f_v = (
                d
                * self.viscosity
                * (self.ps.m[p_j] / (self.ps.density[p_j]))
                * v_xy
                / (r.norm() ** 2 + 0.01 * self.ps.support_radius**2)
                * self.cubic_kernel_derivative(r)
            )
            ret += f_v
        elif self.ps.material[p_j] == self.ps.material_solid:
            boundary_viscosity = 0.0
            # Boundary neighbors
            ## Akinci2012
            f_v = (
                d
                * boundary_viscosity
                * (self.density_0 * self.ps.m_V[p_j] / (self.ps.density[p_i]))
                * v_xy
                / (r.norm() ** 2 + 0.01 * self.ps.support_radius**2)
                * self.cubic_kernel_derivative(r)
            )
            ret += f_v
            if self.ps.is_dynamic_rigid_body(p_j):
                self.ps.acceleration[p_j] += -f_v * self.ps.density[p_i] / self.ps.density[p_j]

    @ti.kernel
    def compute_non_pressure_forces(self):
        for p_i in ti.grouped(self.ps.x):
            if self.ps.is_static_rigid_body(p_i):
                self.ps.acceleration[p_i].fill(0.0)
                continue
            ############## Body force ###############
            # Add body force
            d_v = ti.Vector(self.g)
            self.ps.acceleration[p_i] = d_v
            if self.ps.material[p_i] == self.ps.material_fluid:
                self.ps.for_all_neighbors(p_i, self.compute_non_pressure_forces_task, d_v)
                self.ps.acceleration[p_i] = d_v

    @ti.kernel
    def advect(self):
        # Update position
        for p_i in ti.grouped(self.ps.x):
            if self.ps.is_dynamic[p_i]:
                if self.ps.is_dynamic_rigid_body(p_i):
                    self.ps.v[p_i] += self.dt[None] * self.ps.acceleration[p_i]
                self.ps.x[p_i] += self.dt[None] * self.ps.v[p_i]

    @ti.kernel
    def compute_DFSPH_factor(self):
        for p_i in ti.grouped(self.ps.x):
            if self.ps.material[p_i] != self.ps.material_fluid:
                continue
            sum_grad_p_k = 0.0
            grad_p_i = ti.Vector([0.0 for _ in range(self.ps.dim)])

            # `ret` concatenates `grad_p_i` and `sum_grad_p_k`
            ret = ti.Vector([0.0 for _ in range(self.ps.dim + 1)])

            self.ps.for_all_neighbors(p_i, self.compute_DFSPH_factor_task, ret)

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
            self.ps.dfsph_factor[p_i] = factor

    @ti.func
    def compute_DFSPH_factor_task(self, p_i, p_j, ret: ti.template()):
        if self.ps.material[p_j] == self.ps.material_fluid:
            # Fluid neighbors
            grad_p_j = -self.ps.m_V[p_j] * self.cubic_kernel_derivative(self.ps.x[p_i] - self.ps.x[p_j])
            ret[3] += grad_p_j.norm_sqr()  # sum_grad_p_k
            for i in ti.static(range(3)):  # grad_p_i
                ret[i] -= grad_p_j[i]
        elif self.ps.material[p_j] == self.ps.material_solid:
            # Boundary neighbors
            ## Akinci2012
            grad_p_j = -self.ps.m_V[p_j] * self.cubic_kernel_derivative(self.ps.x[p_i] - self.ps.x[p_j])
            for i in ti.static(range(3)):  # grad_p_i
                ret[i] -= grad_p_j[i]

    @ti.kernel
    def compute_density_change(self):
        for p_i in ti.grouped(self.ps.x):
            if self.ps.material[p_i] != self.ps.material_fluid:
                continue
            ret = ti.Struct(density_adv=0.0, num_neighbors=0)
            self.ps.for_all_neighbors(p_i, self.compute_density_change_task, ret)

            # only correct positive divergence
            density_adv = ti.max(ret.density_adv, 0.0)
            num_neighbors = ret.num_neighbors

            # Do not perform divergence solve when paritlce deficiency happens
            if self.ps.dim == 3:
                if num_neighbors < 20:
                    density_adv = 0.0
            else:
                if num_neighbors < 7:
                    density_adv = 0.0

            self.ps.density_adv[p_i] = density_adv

    @ti.func
    def compute_density_change_task(self, p_i, p_j, ret: ti.template()):
        v_i = self.ps.v[p_i]
        v_j = self.ps.v[p_j]
        if self.ps.material[p_j] == self.ps.material_fluid:
            # Fluid neighbors
            ret.density_adv += self.ps.m_V[p_j] * (v_i - v_j).dot(
                self.cubic_kernel_derivative(self.ps.x[p_i] - self.ps.x[p_j])
            )
        elif self.ps.material[p_j] == self.ps.material_solid:
            # Boundary neighbors
            ## Akinci2012
            ret.density_adv += self.ps.m_V[p_j] * (v_i - v_j).dot(
                self.cubic_kernel_derivative(self.ps.x[p_i] - self.ps.x[p_j])
            )

        # Compute the number of neighbors
        ret.num_neighbors += 1

    @ti.kernel
    def compute_density_adv(self):
        for p_i in ti.grouped(self.ps.x):
            if self.ps.material[p_i] != self.ps.material_fluid:
                continue
            delta = 0.0
            self.ps.for_all_neighbors(p_i, self.compute_density_adv_task, delta)
            density_adv = self.ps.density[p_i] / self.density_0 + self.dt[None] * delta
            self.ps.density_adv[p_i] = ti.max(density_adv, 1.0)

    @ti.func
    def compute_density_adv_task(self, p_i, p_j, ret: ti.template()):
        v_i = self.ps.v[p_i]
        v_j = self.ps.v[p_j]
        if self.ps.material[p_j] == self.ps.material_fluid:
            # Fluid neighbors
            ret += self.ps.m_V[p_j] * (v_i - v_j).dot(self.cubic_kernel_derivative(self.ps.x[p_i] - self.ps.x[p_j]))
        elif self.ps.material[p_j] == self.ps.material_solid:
            # Boundary neighbors
            ## Akinci2012
            ret += self.ps.m_V[p_j] * (v_i - v_j).dot(self.cubic_kernel_derivative(self.ps.x[p_i] - self.ps.x[p_j]))

    @ti.kernel
    def compute_density_error(self, offset: float) -> float:
        density_error = 0.0
        for I in ti.grouped(self.ps.x):
            if self.ps.material[I] == self.ps.material_fluid:
                density_error += self.density_0 * self.ps.density_adv[I] - offset
        return density_error

    @ti.kernel
    def multiply_time_step(self, field: ti.template(), time_step: float):
        for I in ti.grouped(self.ps.x):
            if self.ps.material[I] == self.ps.material_fluid:
                field[I] *= time_step

    def divergence_solve(self):
        # TODO: warm start
        # Compute velocity of density change
        self.compute_density_change()
        inv_dt = 1 / self.dt[None]
        self.multiply_time_step(self.ps.dfsph_factor, inv_dt)

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

        self.multiply_time_step(self.ps.dfsph_factor, self.dt[None])

    def divergence_solver_iteration(self):
        self.divergence_solver_iteration_kernel()
        self.compute_density_change()
        density_err = self.compute_density_error(0.0)
        return density_err / self.ps.fluid_particle_num

    @ti.kernel
    def divergence_solver_iteration_kernel(self):
        # Perform Jacobi iteration
        for p_i in ti.grouped(self.ps.x):
            if self.ps.material[p_i] != self.ps.material_fluid:
                continue
            # evaluate rhs
            b_i = self.ps.density_adv[p_i]
            k_i = b_i * self.ps.dfsph_factor[p_i]
            ret = ti.Struct(dv=ti.Vector([0.0 for _ in range(self.ps.dim)]), k_i=k_i)
            # TODO: if warm start
            # get_kappa_V += k_i
            self.ps.for_all_neighbors(p_i, self.divergence_solver_iteration_task, ret)
            self.ps.v[p_i] += ret.dv

    @ti.func
    def divergence_solver_iteration_task(self, p_i, p_j, ret: ti.template()):
        if self.ps.material[p_j] == self.ps.material_fluid:
            # Fluid neighbors
            b_j = self.ps.density_adv[p_j]
            k_j = b_j * self.ps.dfsph_factor[p_j]
            k_sum = (
                ret.k_i + self.density_0 / self.density_0 * k_j
            )  # TODO: make the neighbor density0 different for multiphase fluid
            if ti.abs(k_sum) > self.m_eps:
                grad_p_j = -self.ps.m_V[p_j] * self.cubic_kernel_derivative(self.ps.x[p_i] - self.ps.x[p_j])
                ret.dv -= self.dt[None] * k_sum * grad_p_j
        elif self.ps.material[p_j] == self.ps.material_solid:
            # Boundary neighbors
            ## Akinci2012
            if ti.abs(ret.k_i) > self.m_eps:
                grad_p_j = -self.ps.m_V[p_j] * self.cubic_kernel_derivative(self.ps.x[p_i] - self.ps.x[p_j])
                vel_change = -self.dt[None] * 1.0 * ret.k_i * grad_p_j
                ret.dv += vel_change
                if self.ps.is_dynamic_rigid_body(p_j):
                    self.ps.acceleration[p_j] += (
                        -vel_change * (1 / self.dt[None]) * self.ps.density[p_i] / self.ps.density[p_j]
                    )

    def pressure_solve(self):
        inv_dt = 1 / self.dt[None]
        inv_dt2 = 1 / (self.dt[None] * self.dt[None])

        # TODO: warm start

        # Compute rho_adv
        self.compute_density_adv()

        self.multiply_time_step(self.ps.dfsph_factor, inv_dt2)

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
        return density_err / self.ps.fluid_particle_num

    @ti.kernel
    def pressure_solve_iteration_kernel(self):
        # Compute pressure forces
        for p_i in ti.grouped(self.ps.x):
            if self.ps.material[p_i] != self.ps.material_fluid:
                continue
            # Evaluate rhs
            b_i = self.ps.density_adv[p_i] - 1.0
            k_i = b_i * self.ps.dfsph_factor[p_i]

            # TODO: if warmstart
            # get kappa V
            self.ps.for_all_neighbors(p_i, self.pressure_solve_iteration_task, k_i)

    @ti.func
    def pressure_solve_iteration_task(self, p_i, p_j, k_i: ti.template()):
        if self.ps.material[p_j] == self.ps.material_fluid:
            # Fluid neighbors
            b_j = self.ps.density_adv[p_j] - 1.0
            k_j = b_j * self.ps.dfsph_factor[p_j]
            k_sum = (
                k_i + self.density_0 / self.density_0 * k_j
            )  # TODO: make the neighbor density0 different for multiphase fluid
            if ti.abs(k_sum) > self.m_eps:
                grad_p_j = -self.ps.m_V[p_j] * self.cubic_kernel_derivative(self.ps.x[p_i] - self.ps.x[p_j])
                # Directly update velocities instead of storing pressure accelerations
                self.ps.v[p_i] -= self.dt[None] * k_sum * grad_p_j  # ki, kj already contain inverse density
        elif self.ps.material[p_j] == self.ps.material_solid:
            # Boundary neighbors
            ## Akinci2012
            if ti.abs(k_i) > self.m_eps:
                grad_p_j = -self.ps.m_V[p_j] * self.cubic_kernel_derivative(self.ps.x[p_i] - self.ps.x[p_j])

                # Directly update velocities instead of storing pressure accelerations
                vel_change = -self.dt[None] * 1.0 * k_i * grad_p_j  # kj already contains inverse density
                self.ps.v[p_i] += vel_change
                if self.ps.is_dynamic_rigid_body(p_j):
                    self.ps.acceleration[p_j] += (
                        -vel_change * 1.0 / self.dt[None] * self.ps.density[p_i] / self.ps.density[p_j]
                    )

    @ti.kernel
    def predict_velocity(self):
        # compute new velocities only considering non-pressure forces
        for p_i in ti.grouped(self.ps.x):
            if self.ps.is_dynamic[p_i] and self.ps.material[p_i] == self.ps.material_fluid:
                self.ps.v[p_i] += self.dt[None] * self.ps.acceleration[p_i]

    def substep(self):
        self.compute_densities()
        self.compute_DFSPH_factor()
        if self.enable_divergence_solver:
            self.divergence_solve()
        self.compute_non_pressure_forces()
        self.predict_velocity()
        self.pressure_solve()
        self.advect()


meta.ps = ParticleSystem(meta.config, GGUI=True)

# ---------------------------------------------------------------------------- #
#                                     main                                     #
# ---------------------------------------------------------------------------- #
def main():
    parser = argparse.ArgumentParser(description="SPH Taichi")
    config = meta.config

    substeps = config.get_cfg("numberOfStepsPerRenderUpdate")

    ps = meta.ps
    solver = ps.build_solver()
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
    x_max, y_max, z_max = config.get_cfg("domainEnd")
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