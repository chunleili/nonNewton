import os
import argparse
import taichi as ti
import numpy as np
import json
import trimesh
import meshio
import plyfile
from dataclasses import dataclass
from functools import reduce
from abc import abstractmethod

ti.init(arch=ti.gpu, device_memory_fraction=0.5)

sph_root_path = os.path.dirname(os.path.abspath(__file__))


class Meta:
    ...


meta = Meta()

FLUID = 0
SOLID = 1

RED = (1, 0, 0)
GREEN = (0, 1, 0)
BLUE = (0, 0, 1)
ORANGE = (1, 0.5, 0)
WHITE = (1, 1, 1)
BLACK = (0, 0, 0)
YELLOW = (1, 1, 0)

STATIC = 0
RIGID = 1
ELASTIC = 2


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


def get_fluid_cfg():
    return meta.config.config.get("FluidParticles", [])


def get_solid_cfg():
    return meta.config.config.get("SolidParticles", [])


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


def read_ply_particles_with_user_data(geometryFile):
    plydata = plyfile.PlyData.read(geometryFile)
    res = plydata["vertex"].data
    return res


def transform(points: np.ndarray, cfg: dict):
    mesh = trimesh.Trimesh(vertices=points)
    rotation_angle = cfg.get("rotationAngle", None)
    rotation_axis = cfg.get("rotationAxis", None)
    if rotation_angle is not None and rotation_axis is not None:
        mesh.apply_transform(
            trimesh.transformations.rotation_matrix(np.deg2rad(rotation_angle), rotation_axis, point=mesh.center_mass)
        )
    trans = cfg.get("translation", None)
    if trans is not None:
        mesh.apply_translation(trans)
    return mesh.vertices


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


@ti.kernel
def range_assign(start: int, end: int, value: ti.template(), container: ti.template()):
    for i in range(start, end):
        container[i] = value


@ti.kernel
def assign(value: ti.template(), container: ti.template()):
    for i in container:
        container[i] = value


@ti.kernel
def range_copy(dst: ti.template(), src: ti.template(), start: int):
    """
    将src的内容复制到 dst 的start开始位置
    """
    for i in src:
        dst[start + i] = src[i]


@ti.kernel
def deep_copy(dst: ti.template(), src: ti.template()):
    for i in src:
        dst[i] = src[i]


def color_selector(cfg, default=WHITE):
    if "color" not in cfg:
        return default
    else:
        res = tuple()
        c = cfg.get("color", "WHITE")
        if c == "RED":
            res = RED
        elif c == "GREEN":
            res = GREEN
        elif c == "BLUE":
            res = BLUE
        elif c == "ORANGE":
            res = ORANGE
        elif c == "WHITE":
            res = WHITE
        elif c == "BLACK":
            res = BLACK
        elif c == "YELLOW":
            res = YELLOW
        else:
            res = WHITE
    return res


def solid_type_selector(cfg, default=STATIC):
    if "solidType" not in cfg:
        return default
    else:
        res = tuple()
        c = cfg.get("solidType")
        if c == "STATIC":
            res = STATIC
        elif c == "RIGID":
            res = RIGID
        elif c == "ELASTIC":
            res = ELASTIC
        else:
            res = STATIC
    return res


@ti.func
def is_static_rigid_body(p):
    return meta.pd.material[p] == SOLID and (not meta.pd.is_dynamic[p])


@ti.func
def is_dynamic_rigid_body(p):
    return meta.pd.material[p] == SOLID and meta.pd.is_dynamic[p]


def animate_particles(pos):
    """
    Use a ply file to animate particles. The format should be:
    frame, dx, dy, dz
    """
    if not hasattr(meta, "anime_file"):
        meta.anime_file = get_cfg("animeFile", None)

    if meta.anime_file is not None:
        pts = read_ply_particles_with_user_data(sph_root_path + meta.anime_file)
        frames = pts["frame"]
        dx, dy, dz = pts["dx"], pts["dy"], pts["dz"]

        for i, f in enumerate(frames):
            if f == int(meta.frame / 1):
                dx_, dy_, dz_ = dx[i], dy[i], dz[i]
                dx_ti = ti.Vector([dx_, dy_, dz_])
                animate_particles_kernel(pos, dx_ti)


@ti.kernel
def animate_particles_kernel(pos: ti.template(), dx_: ti.template()):
    for i in pos:
        pos[i][0] += dx_[0]
        pos[i][1] += dx_[1]
        pos[i][2] += dx_[2]


# ---------------------------------------------------------------------------- #
#                                   RigidBody                                  #
# ---------------------------------------------------------------------------- #
@ti.data_oriented
class RigidBody:
    def __init__(self, extern_pos, phase_id):
        init_pos = extern_pos.to_numpy().copy()

        self.phase_id = phase_id
        self.num_particles = init_pos.shape[0]
        self.dt = meta.parm.dt[None]
        self.gravity = meta.parm.gravity
        self.mass_inv = 1.0
        self.positions = extern_pos
        self.positions0 = ti.Vector.field(3, dtype=ti.f32, shape=self.num_particles)
        self.velocities = ti.Vector.field(3, dtype=ti.f32, shape=self.num_particles)
        self.q_inv = ti.Matrix.field(n=3, m=3, dtype=float, shape=())
        self.radius_vector = ti.Vector.field(3, dtype=ti.f32, shape=self.num_particles)

        self.positions.from_numpy(init_pos)
        self.positions0.from_numpy(init_pos)

        self.compute_radius_vector(self.num_particles, self.positions, self.radius_vector)
        self.precompute_q_inv(self.num_particles, self.radius_vector, self.q_inv)

    def substep(self):
        self.shape_matching(
            self.num_particles,
            self.positions0,
            self.positions,
            self.velocities,
            self.mass_inv,
            self.dt,
            self.q_inv,
            self.radius_vector,
        )

    @ti.kernel
    def shape_matching(
        self,
        num_particles: int,
        positions0: ti.template(),
        positions: ti.template(),
        velocities: ti.template(),
        mass_inv: float,
        dt: ti.f32,
        q_inv: ti.template(),
        radius_vector: ti.template(),
    ):
        # #  update vel and pos firtly
        # gravity = self.gravity
        # for i in range(num_particles):
        #     positions0[i] = positions[i]
        #     f = gravity
        #     velocities[i] += mass_inv * f * dt
        #     positions[i] += velocities[i] * dt
        #     if positions[i].y < 0.0:
        #         positions[i] = positions0[i]
        #         positions[i].y = 0.0

        # compute the new(matched shape) mass center
        c = ti.Vector([0.0, 0.0, 0.0])
        for i in range(num_particles):
            c += positions[i]
        c /= num_particles

        # compute transformation matrix and extract rotation
        sum1 = ti.Matrix([[0.0] * 3 for _ in range(3)], ti.f32)
        for i in range(num_particles):
            sum1 += (positions[i] - c).outer_product(radius_vector[i])
        A = sum1 @ q_inv[None]

        R, _ = ti.polar_decompose(A)

        # update velocities and positions
        for i in range(num_particles):
            positions[i] = c + R @ radius_vector[i]
            # velocities[i] = (positions[i] - positions0[i]) / dt

    @ti.kernel
    def compute_radius_vector(self, num_particles: int, positions: ti.template(), radius_vector: ti.template()):
        # compute the mass center and radius vector
        center_mass = ti.Vector([0.0, 0.0, 0.0])
        for i in range(num_particles):
            center_mass += positions[i]
        center_mass /= num_particles
        for i in range(num_particles):
            radius_vector[i] = positions[i] - center_mass

    @ti.kernel
    def precompute_q_inv(self, num_particles: int, radius_vector: ti.template(), q_inv: ti.template()):
        res = ti.Matrix([[0.0] * 3 for _ in range(3)], ti.f64)
        for i in range(num_particles):
            res += radius_vector[i].outer_product(radius_vector[i])
        q_inv[None] = res.inverse()


@dataclass
class PhaseInfo:
    uid: int = 0
    parnum: int = 0
    startnum: int = 0  # the start index of this phase in the particle array
    material: int = FLUID
    is_dynamic: bool = False
    cfg: dict = None
    color: tuple = WHITE
    pos: np.ndarray = None
    solid_type: int = STATIC


meta.phase_info = dict()


@ti.data_oriented
class Parameter:
    """A pure data class storing parameters for simulation"""

    def __init__(self):
        self.domain_start = np.array(get_cfg("domainStart", [0.0, 0.0, 0.0]))
        self.domain_end = np.array(get_cfg("domainEnd", [1.0, 1.0, 1.0]))
        self.domain_size = self.domain_end - self.domain_start
        self.dim = len(self.domain_size)
        self.simulation_method = get_cfg("simulationMethod")
        self.particle_radius = get_cfg("particleRadius", 0.01)
        self.particle_diameter = 2 * self.particle_radius
        self.support_radius = self.particle_radius * 4.0  # support radius
        self.m_V0 = 0.8 * self.particle_diameter**self.dim
        self.density0 = get_cfg("density0", 1000.0)  # reference density
        self.gravity = ti.Vector(get_cfg("gravitation", [0.0, -9.8, 0.0]))
        self.viscosity = 0.01  # viscosity
        self.boundary_viscosity = get_cfg("boundaryViscosity", 0.0)
        self.dt = ti.field(float, shape=())
        self.dt[None] = get_cfg("timeStepSize", 1e-4)
        self.sticky_coeff = get_cfg("stickyCoefficient", 0.999)
        self.collision_coeff = get_cfg("collisionCoefficient", 0.5)
        self.padding = self.support_radius
        self.coupling_interval = get_cfg("couplingInterval", 1)


# ---------------------------------------------------------------------------- #
#                                load particles                                #
# ---------------------------------------------------------------------------- #
# load all particles info into phase_info, alway first fluid, second static solid, third dynamic solid.
def load_particles():
    cfgs = get_fluid_cfg()
    fluid_particle_num = 0
    cnt_fluid = 0
    for cfg_i in cfgs:
        fluid_particle_num, cnt_fluid = parse_cfg(cfg_i, cnt_fluid, fluid_particle_num, 0, FLUID, STATIC, BLUE, 1)

    cfgs = get_solid_cfg()
    static_par_num = fluid_particle_num
    cnt_static = 0
    for cfg_i in cfgs:
        solid_type = solid_type_selector(cfg_i)
        if solid_type == STATIC:
            static_par_num, cnt_static = parse_cfg(cfg_i, cnt_static, static_par_num, 1000, SOLID, STATIC, WHITE, 0)

    rigid_par_num = static_par_num
    cnt_rigid = 0
    for cfg_i in cfgs:
        solid_type = solid_type_selector(cfg_i)
        if solid_type == RIGID:
            rigid_par_num, cnt_rigid = parse_cfg(cfg_i, cnt_rigid, rigid_par_num, 2000, SOLID, RIGID, ORANGE, 1)

    elastic_par_num = rigid_par_num
    cnt_elastic = 0
    for cfg_i in cfgs:
        solid_type = solid_type_selector(cfg_i)
        if solid_type == ELASTIC:
            elastic_par_num, cnt_elastic = parse_cfg(
                cfg_i, cnt_elastic, elastic_par_num, 3000, SOLID, ELASTIC, YELLOW, 1
            )

    solid_particle_num = elastic_par_num - fluid_particle_num
    particle_max_num = fluid_particle_num + solid_particle_num
    return particle_max_num, fluid_particle_num


def parse_cfg(cfg_i, cnt, startnum, phase_id_start, default_material, default_solid_type, default_color, is_dynamic):
    pos = read_ply_particles(sph_root_path + cfg_i["geometryFile"])
    pos = transform(pos, cfg_i)
    parnum = pos.shape[0]
    phase_id = cfg_i.get("id", cnt) + phase_id_start  # rigid solid phase_id starts from 2000
    cnt += 1
    color = color_selector(cfg_i, default_color)
    meta.phase_info[phase_id] = PhaseInfo(
        uid=phase_id,
        parnum=pos.shape[0],
        startnum=startnum,
        material=default_material,
        cfg=cfg_i,
        is_dynamic=is_dynamic,
        pos=pos,
        color=color,
        solid_type=default_solid_type,
    )
    endnum = startnum + parnum
    return endnum, cnt


# ---------------------------------------------------------------------------- #
#                                 particle data                                #
# ---------------------------------------------------------------------------- #
@ti.data_oriented
class ParticleData:
    """A pure data class for particle data storage and management"""

    def __init__(self, particle_max_num: int):
        self.particle_max_num = particle_max_num

        self.dim = 3

        # Particle related properties
        self.phase_id = ti.field(dtype=int, shape=self.particle_max_num)
        self.x = ti.Vector.field(self.dim, dtype=float, shape=self.particle_max_num)
        self.x_0 = ti.Vector.field(self.dim, dtype=float, shape=self.particle_max_num)
        self.v = ti.Vector.field(self.dim, dtype=float, shape=self.particle_max_num)
        self.acceleration = ti.Vector.field(self.dim, dtype=float, shape=self.particle_max_num)
        self.m_V = ti.field(dtype=float, shape=self.particle_max_num)
        self.m = ti.field(dtype=float, shape=self.particle_max_num)
        self.density = ti.field(dtype=float, shape=self.particle_max_num)
        self.pressure = ti.field(dtype=float, shape=self.particle_max_num)
        self.material = ti.field(dtype=int, shape=self.particle_max_num)
        self.color = ti.Vector.field(3, dtype=ti.f32, shape=self.particle_max_num)
        self.is_dynamic = ti.field(dtype=int, shape=self.particle_max_num)
        self.particle_id = ti.field(dtype=int, shape=self.particle_max_num)

        if get_cfg("simulationMethod") == 4:
            self.dfsph_factor = ti.field(dtype=float, shape=self.particle_max_num)
            self.density_adv = ti.field(dtype=float, shape=self.particle_max_num)

        # Buffer for sort
        self.phase_id_buffer = ti.field(dtype=int, shape=self.particle_max_num)
        self.x_buffer = ti.Vector.field(self.dim, dtype=float, shape=self.particle_max_num)
        self.x_0_buffer = ti.Vector.field(self.dim, dtype=float, shape=self.particle_max_num)
        self.v_buffer = ti.Vector.field(self.dim, dtype=float, shape=self.particle_max_num)
        self.acceleration_buffer = ti.Vector.field(self.dim, dtype=float, shape=self.particle_max_num)
        self.m_V_buffer = ti.field(dtype=float, shape=self.particle_max_num)
        self.m_buffer = ti.field(dtype=float, shape=self.particle_max_num)
        self.density_buffer = ti.field(dtype=float, shape=self.particle_max_num)
        self.pressure_buffer = ti.field(dtype=float, shape=self.particle_max_num)
        self.material_buffer = ti.field(dtype=int, shape=self.particle_max_num)
        self.color_buffer = ti.Vector.field(3, dtype=ti.f32, shape=self.particle_max_num)
        self.is_dynamic_buffer = ti.field(dtype=int, shape=self.particle_max_num)
        self.particle_id_buffer = ti.field(dtype=int, shape=self.particle_max_num)

        if get_cfg("simulationMethod") == 4:
            self.dfsph_factor_buffer = ti.field(dtype=float, shape=self.particle_max_num)
            self.density_adv_buffer = ti.field(dtype=float, shape=self.particle_max_num)


# ---------------------------------------------------------------------------- #
#                             initialize particles                             #
# ---------------------------------------------------------------------------- #
@ti.kernel
def init_particle_id(particle_id: ti.template()):
    for i in particle_id:
        particle_id[i] = i


def initialize_particles(pd: ParticleData):
    """Assign the data into particle data, init those fields"""
    # join the pos arr
    all_par = np.concatenate([phase.pos for phase in meta.phase_info.values()], axis=0)

    pd.x.from_numpy(all_par)
    pd.x_0.from_numpy(all_par)
    pd.m_V.fill(meta.parm.m_V0)
    pd.m.fill(meta.parm.m_V0 * 1000.0)
    pd.density.fill(meta.parm.density0)
    pd.pressure.fill(0.0)
    pd.material.fill(FLUID)
    pd.color.fill(WHITE)
    pd.is_dynamic.fill(1)
    init_particle_id(meta.pd.particle_id)

    for phase in meta.phase_info.values():
        # assign material, color, is_dynamic, phase_id
        start = phase.startnum
        end = start + phase.parnum
        range_assign(start, end, phase.material, pd.material)
        range_assign(start, end, phase.color, pd.color)
        range_assign(start, end, phase.is_dynamic, pd.is_dynamic)
        range_assign(start, end, phase.uid, pd.phase_id)


# ---------------------------------------------------------------------------- #
#                              NeighborhoodSearch                              #
# ---------------------------------------------------------------------------- #
@ti.data_oriented
class NeighborhoodSearch:
    def __init__(self):
        self.particle_max_num = meta.particle_max_num
        self.grid_size = meta.parm.support_radius
        self.grid_num = np.ceil(meta.parm.domain_size / self.grid_size).astype(int)

        # Particle num of each grid
        self.grid_particles_num = ti.field(int, shape=int(self.grid_num[0] * self.grid_num[1] * self.grid_num[2]))
        self.grid_particles_num_temp = ti.field(int, shape=int(self.grid_num[0] * self.grid_num[1] * self.grid_num[2]))
        # Grid id for each particle
        self.grid_ids = ti.field(int, shape=self.particle_max_num)
        self.grid_ids_buffer = ti.field(int, shape=self.particle_max_num)
        self.grid_ids_new = ti.field(int, shape=self.particle_max_num)

        self.prefix_sum_executor = ti.algorithms.PrefixSumExecutor(self.grid_particles_num.shape[0])

        self.max_num_neighbors = 50
        self.neighbors = ti.field(int, shape=(self.particle_max_num, self.max_num_neighbors))
        self.num_neighbors = ti.field(int, shape=self.particle_max_num)

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
    def get_neighbor(self, i, j):
        return self.neighbors[i, j]

    @ti.func
    def get_num_neighbors(self, i):
        return self.num_neighbors[i]

    @ti.kernel
    def update_grid_id(self):
        for I in ti.grouped(self.grid_particles_num):
            self.grid_particles_num[I] = 0
        for I in ti.grouped(meta.pd.x):
            grid_index = self.get_flatten_grid_index(meta.pd.x[I])
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
            # if meta.pd.material[I] != FLUID:
            #     continue
            new_index = self.grid_ids_new[I]
            self.grid_ids_buffer[new_index] = self.grid_ids[I]
            meta.pd.phase_id_buffer[new_index] = meta.pd.phase_id[I]
            meta.pd.x_0_buffer[new_index] = meta.pd.x_0[I]
            meta.pd.x_buffer[new_index] = meta.pd.x[I]
            meta.pd.v_buffer[new_index] = meta.pd.v[I]
            meta.pd.acceleration_buffer[new_index] = meta.pd.acceleration[I]
            meta.pd.m_V_buffer[new_index] = meta.pd.m_V[I]
            meta.pd.m_buffer[new_index] = meta.pd.m[I]
            meta.pd.density_buffer[new_index] = meta.pd.density[I]
            meta.pd.pressure_buffer[new_index] = meta.pd.pressure[I]
            meta.pd.material_buffer[new_index] = meta.pd.material[I]
            meta.pd.color_buffer[new_index] = meta.pd.color[I]
            meta.pd.is_dynamic_buffer[new_index] = meta.pd.is_dynamic[I]
            meta.pd.particle_id_buffer[new_index] = meta.pd.particle_id[I]

            if ti.static(meta.parm.simulation_method == 4):
                meta.pd.dfsph_factor_buffer[new_index] = meta.pd.dfsph_factor[I]
                meta.pd.density_adv_buffer[new_index] = meta.pd.density_adv[I]

        for I in ti.grouped(meta.pd.x):
            # if meta.pd.material[I] != FLUID:
            #     continue
            self.grid_ids[I] = self.grid_ids_buffer[I]
            meta.pd.phase_id[I] = meta.pd.phase_id_buffer[I]
            meta.pd.x_0[I] = meta.pd.x_0_buffer[I]
            meta.pd.x[I] = meta.pd.x_buffer[I]
            meta.pd.v[I] = meta.pd.v_buffer[I]
            meta.pd.acceleration[I] = meta.pd.acceleration_buffer[I]
            meta.pd.m_V[I] = meta.pd.m_V_buffer[I]
            meta.pd.m[I] = meta.pd.m_buffer[I]
            meta.pd.density[I] = meta.pd.density_buffer[I]
            meta.pd.pressure[I] = meta.pd.pressure_buffer[I]
            meta.pd.material[I] = meta.pd.material_buffer[I]
            meta.pd.color[I] = meta.pd.color_buffer[I]
            meta.pd.is_dynamic[I] = meta.pd.is_dynamic_buffer[I]
            meta.pd.particle_id[I] = meta.pd.particle_id_buffer[I]

            if ti.static(meta.parm.simulation_method == 4):
                meta.pd.dfsph_factor[I] = meta.pd.dfsph_factor_buffer[I]
                meta.pd.density_adv[I] = meta.pd.density_adv_buffer[I]

    def run_search(self):
        self.update_grid_id()
        self.prefix_sum_executor.run(self.grid_particles_num)
        self.counting_sort()

        self.num_neighbors.fill(0)
        self.neighbors.fill(-1)
        self.store_neighbors()

    @ti.func
    def for_all_neighbors(self, p_i, task: ti.template(), ret: ti.template()):
        center_cell = self.pos_to_index(meta.pd.x[p_i])
        for offset in ti.grouped(ti.ndrange(*((-1, 2),) * meta.parm.dim)):
            grid_index = self.flatten_grid_index(center_cell + offset)
            for p_j in range(self.grid_particles_num[ti.max(0, grid_index - 1)], self.grid_particles_num[grid_index]):
                if p_i[0] != p_j and (meta.pd.x[p_i] - meta.pd.x[p_j]).norm() < meta.parm.support_radius:
                    task(p_i, p_j, ret)

    # this is proved same, but I prefer the below one
    # @ti.kernel
    # def store_neighbors(self):
    #     for p_i in ti.grouped(meta.pd.x):
    #         num_neighbors=0
    #         self.for_all_neighbors(p_i, self.store_neighbors_task, num_neighbors)

    # @ti.func
    # def store_neighbors_task(self, p_i, p_j, ret: ti.template()):
    #     self.neighbors[p_i, self.num_neighbors[p_i]] = p_j
    #     self.num_neighbors[p_i] += 1

    @ti.kernel
    def store_neighbors(self):
        for p_i in range(self.particle_max_num):
            center_cell = self.pos_to_index(meta.pd.x[p_i])
            for offset in ti.grouped(ti.ndrange(*((-1, 2),) * meta.parm.dim)):
                grid_index = self.flatten_grid_index(center_cell + offset)
                for p_j in range(
                    self.grid_particles_num[ti.max(0, grid_index - 1)], self.grid_particles_num[grid_index]
                ):
                    if p_i != p_j and (meta.pd.x[p_i] - meta.pd.x[p_j]).norm() < meta.parm.support_radius:
                        self.neighbors[p_i, self.num_neighbors[p_i]] = p_j
                        self.num_neighbors[p_i] += 1


# ---------------------------------------------------------------------------- #
#                             fluid solid coupling                             #
# ---------------------------------------------------------------------------- #
# @ti.kernel
# def copy_pos_to_system(src_pos:ti.template(), x:ti.template(), particle_id:ti.template(), startnum:int, endnum:int):
#     for i in range(startnum, endnum):
#             x[particle_id[i]] = src_pos[i-startnum]


@ti.kernel
def copy_pos_to_system(src_pos: ti.template(), x: ti.template(), particle_id: ti.template(), startnum: int):
    for i in range(src_pos.shape[0]):
        src_id = i + startnum
        dst_id = particle_id[src_id]
        x[dst_id] = src_pos[i]


# just copy the rigid body position to the particle system
def oneway_coupling(rb):
    rb_phase_id = rb.phase_id
    startnum = meta.phase_info[rb_phase_id].startnum
    parnum = meta.phase_info[rb_phase_id].parnum
    endnum = startnum + parnum
    copy_pos_to_system(rb.positions, meta.pd.x, meta.pd.particle_id, startnum)
    # range_copy(meta.pd.x, rb.positions, 0)


# ---------------------------------------------------------------------------- #
#                                   SPH Base                                   #
# ---------------------------------------------------------------------------- #
@ti.data_oriented
class SPHBase:
    def __init__(self):
        self.gravity = meta.parm.gravity
        self.viscosity = meta.parm.viscosity
        self.density_0 = meta.parm.density0
        self.dt = meta.parm.dt

    def step(self):
        print("step: ", meta.step_num)
        meta.ns.run_search()
        self.compute_moving_boundary_volume()

        assign(self.gravity, meta.pd.acceleration)
        if meta.fluid_particle_num > 0:
            self.substep()

        # if meta.step_num % meta.parm.coupling_interval == 0:
        for rb in meta.rbs:
            for i in range(meta.parm.coupling_interval):
                rb.substep()
                self.enforce_boundary_3D(meta.pd.x, meta.pd.v, SOLID)

                # oneway_coupling(rb)
                # print("rb: ", rb.phase_id, "step: ", meta.step_num)

        self.advect()

        animate_particles(meta.pd.x)
        self.enforce_boundary_3D(meta.pd.x, meta.pd.v, FLUID)

    @abstractmethod
    def substep(self):
        pass

    def initialize(self):
        meta.ns.run_search()
        self.compute_static_boundary_volume()
        self.compute_moving_boundary_volume()

    @ti.func
    def cubic_kernel(self, r_norm):
        res = ti.cast(0.0, ti.f32)
        h = meta.parm.support_radius
        # value of cubic spline smoothing kernel
        k = 1.0
        if meta.parm.dim == 1:
            k = 4 / 3
        elif meta.parm.dim == 2:
            k = 40 / 7 / np.pi
        elif meta.parm.dim == 3:
            k = 8 / np.pi
        k /= h**meta.parm.dim
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
        h = meta.parm.support_radius
        # derivative of cubic spline smoothing kernel
        k = 1.0
        if meta.parm.dim == 1:
            k = 4 / 3
        elif meta.parm.dim == 2:
            k = 40 / 7 / np.pi
        elif meta.parm.dim == 3:
            k = 8 / np.pi
        k = 6.0 * k / h**meta.parm.dim
        r_norm = r.norm()
        q = r_norm / h
        res = ti.Vector([0.0 for _ in range(meta.parm.dim)])
        if r_norm > 1e-5 and q <= 1.0:
            grad_q = r / (r_norm * h)
            if q <= 0.5:
                res = k * q * (3.0 * q - 2.0) * grad_q
            else:
                factor = 1.0 - q
                res = k * (-factor * factor) * grad_q
        return res

    @ti.func
    def compute_densities_task(self, p_i, p_j, ret: ti.template()):
        x_i = meta.pd.x[p_i]
        if meta.pd.material[p_j] == FLUID:
            # Fluid neighbors
            x_j = meta.pd.x[p_j]
            ret += meta.pd.m_V[p_j] * self.cubic_kernel((x_i - x_j).norm())
        elif meta.pd.material[p_j] == SOLID:
            # Boundary neighbors
            ## Akinci2012
            x_j = meta.pd.x[p_j]
            ret += meta.pd.m_V[p_j] * self.cubic_kernel((x_i - x_j).norm())

    @ti.kernel
    def compute_densities(self):
        for p_i in ti.grouped(meta.pd.x):
            if meta.pd.material[p_i] != FLUID:
                continue
            meta.pd.density[p_i] = meta.pd.m_V[p_i] * self.cubic_kernel(0.0)
            den = 0.0
            meta.ns.for_all_neighbors(p_i, self.compute_densities_task, den)
            meta.pd.density[p_i] += den
            meta.pd.density[p_i] *= self.density_0

    # this is proved same, but let us keep it for now
    # @ti.kernel
    # def compute_densities_new(self):
    #     for p_i in ti.grouped(meta.pd.x):
    #         if meta.pd.material[p_i] != FLUID:
    #             continue
    #         meta.pd.density[p_i] = meta.pd.m_V[p_i] * self.cubic_kernel(0.0)
    #         den = 0.0
    #         num_neighbors = meta.ns.get_num_neighbors(p_i)
    #         for k in range(num_neighbors):
    #             p_j = meta.ns.get_neighbor(p_i, k)
    #             x_i = meta.pd.x[p_i]
    #             if meta.pd.material[p_j] == FLUID:
    #                 # Fluid neighbors
    #                 x_j = meta.pd.x[p_j]
    #                 den += meta.pd.m_V[p_j] * self.cubic_kernel((x_i - x_j).norm())
    #             elif meta.pd.material[p_j] == SOLID:
    #                 # Boundary neighbors
    #                 ## Akinci2012
    #                 x_j = meta.pd.x[p_j]
    #                 den += meta.pd.m_V[p_j] * self.cubic_kernel((x_i - x_j).norm())
    #         meta.pd.density[p_i] += den
    #         meta.pd.density[p_i] *= self.density_0

    @ti.func
    def viscosity_force(self, p_i, p_j, r):
        # Compute the viscosity force contribution
        v_xy = (meta.pd.v[p_i] - meta.pd.v[p_j]).dot(r)
        res = (
            2
            * (meta.parm.dim + 2)
            * self.viscosity
            * (meta.pd.m[p_j] / (meta.pd.density[p_j]))
            * v_xy
            / (r.norm() ** 2 + 0.01 * meta.parm.support_radius**2)
            * self.cubic_kernel_derivative(r)
        )
        return res

    @ti.kernel
    def compute_static_boundary_volume(self):
        for p_i in ti.grouped(meta.pd.x):
            if not is_static_rigid_body(p_i):
                continue
            delta = self.cubic_kernel(0.0)
            meta.ns.for_all_neighbors(p_i, self.compute_boundary_volume_task, delta)
            meta.pd.m_V[p_i] = (
                1.0 / delta * 3.0
            )  # TODO: the 3.0 here is a coefficient for missing particles by trail and error... need to figure out how to determine it sophisticatedly

    @ti.func
    def compute_boundary_volume_task(self, p_i, p_j, delta: ti.template()):
        if meta.pd.material[p_j] == SOLID:
            delta += self.cubic_kernel((meta.pd.x[p_i] - meta.pd.x[p_j]).norm())

    @ti.kernel
    def compute_moving_boundary_volume(self):
        for p_i in ti.grouped(meta.pd.x):
            if not is_dynamic_rigid_body(p_i):
                continue
            delta = self.cubic_kernel(0.0)
            meta.ns.for_all_neighbors(p_i, self.compute_boundary_volume_task, delta)
            meta.pd.m_V[p_i] = (
                1.0 / delta * 3.0
            )  # TODO: the 3.0 here is a coefficient for missing particles by trail and error... need to figure out how to determine it sophisticatedly

    @ti.func
    def simulate_collisions(self, p_i, normal, vel):
        # Collision factor, assume roughly (1-c_f)*velocity loss after collision
        c_f = meta.parm.collision_coeff  # ~0.5
        vel[p_i] -= (1.0 + c_f) * vel[p_i].dot(normal) * normal
        vel[p_i] *= meta.parm.sticky_coeff  # sticky wall: ~0.999

    @ti.kernel
    def enforce_boundary_3D(self, positions: ti.template(), vel: ti.template(), particle_type: int):
        xmax = meta.parm.domain_end[0] - meta.parm.padding
        ymax = meta.parm.domain_end[1] - meta.parm.padding
        zmax = meta.parm.domain_end[2] - meta.parm.padding
        xmin = meta.parm.domain_start[0] + meta.parm.padding
        ymin = meta.parm.domain_start[1] + meta.parm.padding
        zmin = meta.parm.domain_start[2] + meta.parm.padding

        for p_i in ti.grouped(positions):
            if meta.pd.is_dynamic[p_i] and meta.pd.material[p_i] == particle_type:
                pos = positions[p_i]
                collision_normal = ti.Vector([0.0, 0.0, 0.0])
                if pos[0] > xmax:
                    collision_normal[0] += 1.0
                    positions[p_i][0] = xmax
                if pos[0] <= xmin:
                    collision_normal[0] += -1.0
                    positions[p_i][0] = xmin

                if pos[1] > ymax:
                    collision_normal[1] += 1.0
                    positions[p_i][1] = ymax
                if pos[1] <= ymin:
                    collision_normal[1] += -1.0
                    positions[p_i][1] = ymin

                if pos[2] > zmax:
                    collision_normal[2] += 1.0
                    positions[p_i][2] = zmax
                if pos[2] <= zmin:
                    collision_normal[2] += -1.0
                    positions[p_i][2] = zmin

                collision_normal_length = collision_normal.norm()
                if collision_normal_length > 1e-6:
                    self.simulate_collisions(p_i, collision_normal / collision_normal_length, vel)

    @ti.func
    def grad_w_ij(self, p_i: int, p_j: int):
        return self.cubic_kernel_derivative(meta.pd.x[p_i] - meta.pd.x[p_j])

    @ti.kernel
    def advect(self):
        # Update position
        for p_i in ti.grouped(meta.pd.x):
            if meta.pd.is_dynamic[p_i]:
                if is_dynamic_rigid_body(p_i):
                    meta.pd.v[p_i] += self.dt[None] * meta.pd.acceleration[p_i]
                meta.pd.x[p_i] += self.dt[None] * meta.pd.v[p_i]


# ---------------------------------------------------------------------------- #
#                                  Rigid body                                  #
# ---------------------------------------------------------------------------- #
class RigidBodySolver(SPHBase):
    def __init__(self, phase_id):
        super().__init__()
        self.phase_id = phase_id
        self.rigid_rest_cm = ti.Vector.field(3, dtype=ti.f32, shape=())
        self.compute_rigid_rest_cm(phase_id)
        deep_copy(meta.pd.x_0, meta.pd.x)

    def substep(self):
        self.solve_constraints(self.phase_id)
        # self.enforce_boundary_3D(meta.pd.x, meta.pd.v, SOLID)

    @ti.func
    def compute_com(self, phase_id):
        sum_m = 0.0
        cm = ti.Vector([0.0, 0.0, 0.0])
        for p_i in range(meta.particle_max_num):
            if is_dynamic_rigid_body(p_i) and meta.pd.phase_id[p_i] == phase_id:
                mass = meta.parm.m_V0 * meta.pd.density[p_i]
                cm += mass * meta.pd.x[p_i]
                sum_m += mass
        cm /= sum_m
        return cm

    @ti.kernel
    def compute_rigid_rest_cm(self, phase_id: int):
        self.rigid_rest_cm[None] = self.compute_com(phase_id)

    @ti.kernel
    def solve_constraints(self, phase_id: int) -> ti.types.matrix(3, 3, float):
        # compute center of mass
        cm = self.compute_com(phase_id)
        # A
        A = ti.Matrix([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
        for p_i in range(meta.particle_max_num):
            if is_dynamic_rigid_body(p_i) and meta.pd.phase_id[p_i] == phase_id:
                q = meta.pd.x_0[p_i] - self.rigid_rest_cm[None]
                p = meta.pd.x[p_i] - cm
                A += meta.parm.m_V0 * meta.pd.density[p_i] * p.outer_product(q)

        R, S = ti.polar_decompose(A)

        if all(abs(R) < 1e-6):
            R = ti.Matrix.identity(ti.f32, 3)

        for p_i in range(meta.particle_max_num):
            if is_dynamic_rigid_body(p_i) and meta.pd.phase_id[p_i] == phase_id:
                goal = cm + R @ (meta.pd.x_0[p_i] - self.rigid_rest_cm[None])
                corr = (goal - meta.pd.x[p_i]) * 1.0
                meta.pd.x[p_i] += corr
        return R


# ---------------------------------------------------------------------------- #
#                                   elasticity                                  #
# ---------------------------------------------------------------------------- #
vec6 = ti.types.vector(6, ti.f32)
mat6 = ti.types.matrix(6, 6, ti.f32)
vec3 = ti.types.vector(3, ti.f32)


class Elasticity(SPHBase):
    """elasticity(Becker2009)"""

    def __init__(self, numParticles) -> None:
        self.m_youngsModulus = 1e5
        self.m_poissonRatio = 0.3

        self.numParticles = numParticles
        self.max_num_neighbors = 50
        #  initial particle indices, used to access their original positions
        self.m_current_to_initial_index = ti.field(shape=(self.numParticles,), dtype=ti.i32)
        self.m_initial_to_current_index = ti.field(shape=(self.numParticles,), dtype=ti.i32)
        #  initial particle neighborhood
        self.m_initialNeighbors = ti.field(shape=(self.numParticles, self.max_num_neighbors), dtype=ti.i32)
        self.m_numInitialNeighbors = ti.field(shape=(self.numParticles,), dtype=ti.i32)
        # // volumes in rest configuration
        self.m_restVolumes = ti.field(dtype=ti.f32, shape=self.numParticles)
        self.m_rotations = ti.Matrix.field(3, 3, dtype=ti.f32, shape=self.numParticles)
        self.m_stress = ti.Vector.field(6, dtype=ti.f32, shape=self.numParticles)
        self.m_F = ti.Matrix.field(3, 3, dtype=ti.f32, shape=self.numParticles)
        self.m_alpha = ti.field(dtype=ti.f32, shape=())
        self.m_alpha[None] = 0.0

        self.initValues()

    def initValues(self):
        self.compute_densities()
        self.initValues_kernel()

    @ti.kernel
    def initValues_kernel(self):
        # // Store the neighbors in the reference configurations and
        # // compute the volume of each particle in rest state
        for i in range(self.numParticles):
            self.m_current_to_initial_index[i] = i
            self.m_initial_to_current_index[i] = i

            # only neighbors in same phase will influence elasticity
            numNeighbors = meta.ns.get_num_neighbors(i)
            self.m_numInitialNeighbors = numNeighbors
            for j in range(numNeighbors):
                self.m_initialNeighbors[i, j] = meta.ns.get_neighbor(i, j)

            # // Compute volume
            # density = compute_density(i, density)
            self.m_restVolumes[i] = meta.pd.m[i] / meta.pd.density[i]

            # // mark all particles in the bounding box as fixed
            # determineFixedParticles()

    def step(self):
        self.computeRotations()
        self.computeStress()
        self.computeForces()

    def computeRotations(self):
        self.computeRotations_kernel()

    @ti.kernel
    def computeRotations_kernel(self):
        for i in range(self.numParticles):
            i0 = self.m_current_to_initial_index[i]
            xi = meta.pd.x[i]
            xi0 = meta.pd.x_0[i0]
            Apq = ti.math.mat3(0.0)

            numNeighbors = self.m_numInitialNeighbors[i0]

            # ---------------------------------------------------------------------------- #
            #                                     Fluid                                    #
            # ---------------------------------------------------------------------------- #
            for j in range(numNeighbors):
                neighborIndex = self.m_initial_to_current_index[self.m_initialNeighbors[i0][j]]
                neighborIndex0 = self.m_initialNeighbors[i0][j]

                xj = meta.pd.x[neighborIndex]
                xj0 = meta.pd.x_0[neighborIndex0]
                xj_xi = xj - xi
                xj_xi_0 = xj0 - xi0
                Apq += meta.pd.m[0] * self.cubic_kernel(xj_xi_0) * (xj_xi * xj_xi_0.transpose())

            # extract rotations
            R, _ = ti.polar_decompose(Apq)
            self.m_rotations[i] = R

    def computeStress(self):
        mat6 = ti.types.matrix(6, 6, ti.f32)
        C = mat6(0.0)
        factor = self.m_youngsModulus / ((1.0 + self.m_poissonRatio) * (1.0 - 2.0 * self.m_poissonRatio))
        C[0, 0] = C[1, 1] = C[2, 2] = factor * (1.0 - self.m_poissonRatio)
        C[0, 1] = C[0, 2] = C[1, 0] = C[1, 2] = C[2, 0] = C[2, 1] = factor * (self.m_poissonRatio)
        C[3, 3] = C[4, 4] = C[5, 5] = factor * 0.5 * (1.0 - 2.0 * self.m_poissonRatio)

        self.computeStress_kernel(C)

    @ti.kernel
    def computeStress_kernel(self, C: ti.template()):
        for i in range(self.numParticles):
            i0 = self.m_current_to_initial_index[i]
            xi = meta.pd.x[i]
            xi0 = meta.pd.x_0[i0]

            nablaU = ti.math.mat3(0.0)
            numNeighbors = self.m_numInitialNeighbors[i0]

            # ---------------------------------------------------------------------------- #
            #                                     Fluid                                    #
            # ---------------------------------------------------------------------------- #
            for j in range(numNeighbors):
                neighborIndex = self.m_initial_to_current_index[self.m_initialNeighbors[i0][j]]
                # get initial neighbor index considering the current particle order
                neighborIndex0 = self.m_initialNeighbors[i0][j]
                xj = meta.pd.x[neighborIndex]
                xj0 = meta.pd.x_0[neighborIndex0]

                xj_xi = xj - xi
                xj_xi_0 = xj0 - xi0

                uji = self.m_rotations[i].transpose() @ xj_xi - xj_xi_0
                # subtract because kernel gradient is taken in direction of xji0 instead of xij0
                nablaU -= (self.m_restVolumes[neighborIndex] * uji) * self.cubic_kernel_derivative(xj_xi_0).transpose()
            self.m_F[i] = nablaU + ti.math.eye(3)

            # compute Cauchy strain: epsilon = 0.5 (nabla u + nabla u^T)
            vec6 = ti.types.vector(6, ti.f32)
            strain = vec6(0.0)
            strain[0] = nablaU[0, 0]  # \epsilon_{00}
            strain[1] = nablaU[1, 1]  # \epsilon_{11}
            strain[2] = nablaU[2, 2]  # \epsilon_{22}
            strain[3] = 0.5 * (nablaU[0, 1] + nablaU[1, 0])  # \epsilon_{01}
            strain[4] = 0.5 * (nablaU[0, 2] + nablaU[2, 0])  # \epsilon_{02}
            strain[5] = 0.5 * (nablaU[1, 2] + nablaU[2, 1])  # \epsilon_{12}

            # stress = C * epsilon
            self.m_stress[i] = C @ strain

    def computeForces(self):
        self.computeForces_kernel()

    @ti.func
    def symMatTimesVec(self, M: vec6, v: vec3, res: ti.template()):
        res[0] = M[0] * v[0] + M[3] * v[1] + M[4] * v[2]
        res[1] = M[3] * v[0] + M[1] * v[1] + M[5] * v[2]
        res[2] = M[4] * v[0] + M[5] * v[1] + M[2] * v[2]

    @ti.kernel
    def computeForces_kernel(self):
        for i in range(self.numParticles):
            i0 = self.m_current_to_initial_index[i]
            xi0 = meta.pd.x_0[i0]

            numNeighbors = self.m_numInitialNeighbors[i0]
            fi = ti.math.vec3(0.0)

            # ---------------------------------------------------------------------------- #
            #                                     Fluid                                    #
            # ---------------------------------------------------------------------------- #
            for j in range(numNeighbors):
                neighborIndex = self.m_initial_to_current_index[self.m_initialNeighbors[i0][j]]
                # get initial neighbor index considering the current particle order
                neighborIndex0 = self.m_initialNeighbors[i0][j]

                xj0 = meta.pd.x_0[neighborIndex0]

                xj_xi_0 = xj0 - xi0
                gradW0 = self.cubic_kernel_derivative(xj_xi_0)

                dji = self.m_restVolumes[i] * gradW0
                dij = -self.m_restVolumes[neighborIndex] * gradW0

                sdji = ti.math.vec3(0.0)
                sdij = ti.math.vec3(0.0)

                self.symMatTimesVec(self.m_stress[neighborIndex], dji, sdji)
                self.symMatTimesVec(self.m_stress[i], dij, sdij)

                fij = -self.m_restVolumes[neighborIndex] * sdji
                fji = -self.m_restVolumes[i] * sdij

                fi += self.m_rotations[neighborIndex] @ fij - self.m_rotations[i] @ fji

            fi = 0.5 * fi

            # if (m_alpha != 0.0)
            # Ganzenmüller 2015: NOT IMPLEMENTED YET

            meta.pd.acceleration[i] += fi / meta.pd.m[i]


# ---------------------------------------------------------------------------- #
#                                     DFSPH                                    #
# ---------------------------------------------------------------------------- #
class DFSPHSolver(SPHBase):
    def __init__(self):
        super().__init__()

        self.surface_tension = 0.01
        self.enable_divergence_solver = True
        self.m_max_iterations_v = get_cfg("maxIterationsV", 100)  # max iter for divergence solve
        self.m_max_iterations = get_cfg("maxIterations", 100)  # max iter for pressure solve
        self.m_eps = 1e-5
        self.max_error_V = get_cfg("maxErrorV", 0.1)  # max error of divergence solver iteration in percentage
        self.max_error = get_cfg("maxError", 0.05)  # max error of pressure solve iteration in percentage

    @ti.func
    def compute_non_pressure_forces_task(self, p_i, p_j, ret: ti.template()):
        x_i = meta.pd.x[p_i]

        ############## Surface Tension ###############
        if meta.pd.material[p_j] == FLUID:
            # Fluid neighbors
            diameter2 = meta.parm.particle_diameter * meta.parm.particle_diameter
            x_j = meta.pd.x[p_j]
            r = x_i - x_j
            r2 = r.dot(r)
            if r2 > diameter2:
                ret -= self.surface_tension / meta.pd.m[p_i] * meta.pd.m[p_j] * r * self.cubic_kernel(r.norm())
            else:
                ret -= (
                    self.surface_tension
                    / meta.pd.m[p_i]
                    * meta.pd.m[p_j]
                    * r
                    * self.cubic_kernel(ti.Vector([meta.parm.particle_diameter, 0.0, 0.0]).norm())
                )

        ############### Viscosoty Force ###############
        d = 2 * (meta.parm.dim + 2)
        x_j = meta.pd.x[p_j]
        # Compute the viscosity force contribution
        r = x_i - x_j
        v_xy = (meta.pd.v[p_i] - meta.pd.v[p_j]).dot(r)

        if meta.pd.material[p_j] == FLUID:
            f_v = (
                d
                * self.viscosity
                * (meta.pd.m[p_j] / (meta.pd.density[p_j]))
                * v_xy
                / (r.norm() ** 2 + 0.01 * meta.parm.support_radius**2)
                * self.cubic_kernel_derivative(r)
            )
            ret += f_v
        elif meta.pd.material[p_j] == SOLID:
            if meta.parm.boundary_viscosity != 0.0:
                # Boundary neighbors
                ## Akinci2012
                f_v = (
                    d
                    * meta.parm.boundary_viscosity
                    * (self.density_0 * meta.pd.m_V[p_j] / (meta.pd.density[p_i]))
                    * v_xy
                    / (r.norm() ** 2 + 0.01 * meta.parm.support_radius**2)
                    * self.cubic_kernel_derivative(r)
                )
                ret += f_v
                if is_dynamic_rigid_body(p_j):
                    meta.pd.acceleration[p_j] += -f_v

    @ti.kernel
    def compute_non_pressure_forces(self):
        for p_i in ti.grouped(meta.pd.x):
            if is_static_rigid_body(p_i):
                meta.pd.acceleration[p_i].fill(0.0)
                continue
            ############## Body force ###############
            # Add body force
            d_v = self.gravity
            meta.pd.acceleration[p_i] = d_v
            if meta.pd.material[p_i] == FLUID:
                meta.ns.for_all_neighbors(p_i, self.compute_non_pressure_forces_task, d_v)
                meta.pd.acceleration[p_i] = d_v

    @ti.kernel
    def compute_DFSPH_factor(self):
        for p_i in ti.grouped(meta.pd.x):
            if meta.pd.material[p_i] != FLUID:
                continue
            sum_grad_p_k = 0.0
            grad_p_i = ti.Vector([0.0 for _ in range(meta.parm.dim)])

            # `ret` concatenates `grad_p_i` and `sum_grad_p_k`
            ret = ti.Vector([0.0 for _ in range(meta.parm.dim + 1)])

            meta.ns.for_all_neighbors(p_i, self.compute_DFSPH_factor_task, ret)

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
            meta.pd.dfsph_factor[p_i] = factor

    @ti.func
    def compute_DFSPH_factor_task(self, p_i, p_j, ret: ti.template()):
        if meta.pd.material[p_j] == FLUID:
            # Fluid neighbors
            grad_p_j = -meta.pd.m_V[p_j] * self.grad_w_ij(p_i, p_j)
            ret[3] += grad_p_j.norm_sqr()  # sum_grad_p_k
            for i in ti.static(range(3)):  # grad_p_i
                ret[i] -= grad_p_j[i]
        elif meta.pd.material[p_j] == SOLID:
            # Boundary neighbors
            ## Akinci2012
            grad_p_j = -meta.pd.m_V[p_j] * self.grad_w_ij(p_i, p_j)
            for i in ti.static(range(3)):  # grad_p_i
                ret[i] -= grad_p_j[i]

    @ti.kernel
    def compute_density_change(self):
        for p_i in ti.grouped(meta.pd.x):
            if meta.pd.material[p_i] != FLUID:
                continue
            ret = ti.Struct(density_adv=0.0, num_neighbors=0)
            meta.ns.for_all_neighbors(p_i, self.compute_density_change_task, ret)

            # only correct positive divergence
            density_adv = ti.max(ret.density_adv, 0.0)
            num_neighbors = ret.num_neighbors

            # Do not perform divergence solve when paritlce deficiency happens
            if meta.parm.dim == 3:
                if num_neighbors < 20:
                    density_adv = 0.0
            else:
                if num_neighbors < 7:
                    density_adv = 0.0

            meta.pd.density_adv[p_i] = density_adv

    @ti.func
    def compute_density_change_task(self, p_i, p_j, ret: ti.template()):
        v_i = meta.pd.v[p_i]
        v_j = meta.pd.v[p_j]
        if meta.pd.material[p_j] == FLUID:
            # Fluid neighbors
            ret.density_adv += meta.pd.m_V[p_j] * (v_i - v_j).dot(self.grad_w_ij(p_i, p_j))
        elif meta.pd.material[p_j] == SOLID:
            # Boundary neighbors
            ## Akinci2012
            ret.density_adv += meta.pd.m_V[p_j] * (v_i - v_j).dot(self.grad_w_ij(p_i, p_j))

        # Compute the number of neighbors
        ret.num_neighbors += 1

    @ti.kernel
    def compute_density_adv(self):
        for p_i in ti.grouped(meta.pd.x):
            if meta.pd.material[p_i] != FLUID:
                continue
            delta = 0.0
            meta.ns.for_all_neighbors(p_i, self.compute_density_adv_task, delta)
            density_adv = meta.pd.density[p_i] / self.density_0 + self.dt[None] * delta
            meta.pd.density_adv[p_i] = ti.max(density_adv, 1.0)

    @ti.func
    def compute_density_adv_task(self, p_i, p_j, ret: ti.template()):
        v_i = meta.pd.v[p_i]
        v_j = meta.pd.v[p_j]
        if meta.pd.material[p_j] == FLUID:
            # Fluid neighbors
            ret += meta.pd.m_V[p_j] * (v_i - v_j).dot(self.grad_w_ij(p_i, p_j))
        elif meta.pd.material[p_j] == SOLID:
            # Boundary neighbors
            ## Akinci2012
            ret += meta.pd.m_V[p_j] * (v_i - v_j).dot(self.grad_w_ij(p_i, p_j))

    @ti.kernel
    def compute_density_error(self, offset: float) -> float:
        density_error = 0.0
        for I in ti.grouped(meta.pd.x):
            if meta.pd.material[I] == FLUID:
                density_error += self.density_0 * meta.pd.density_adv[I] - offset
        return density_error

    @ti.kernel
    def multiply_time_step(self, field: ti.template(), time_step: float):
        for I in ti.grouped(meta.pd.x):
            if meta.pd.material[I] == FLUID:
                field[I] *= time_step

    def divergence_solve(self):
        # TODO: warm start
        # Compute velocity of density change
        self.compute_density_change()
        inv_dt = 1.0 / self.dt[None]
        self.multiply_time_step(meta.pd.dfsph_factor, inv_dt)

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

        self.multiply_time_step(meta.pd.dfsph_factor, self.dt[None])

    def divergence_solver_iteration(self):
        self.divergence_solver_iteration_kernel()
        self.compute_density_change()
        density_err = self.compute_density_error(0.0)
        return density_err / meta.fluid_particle_num

    @ti.kernel
    def divergence_solver_iteration_kernel(self):
        # Perform Jacobi iteration
        for p_i in ti.grouped(meta.pd.x):
            if meta.pd.material[p_i] != FLUID:
                continue
            # evaluate rhs
            b_i = meta.pd.density_adv[p_i]
            k_i = b_i * meta.pd.dfsph_factor[p_i]
            ret = ti.Struct(dv=ti.Vector([0.0 for _ in range(meta.parm.dim)]), k_i=k_i)
            # TODO: if warm start
            # get_kappa_V += k_i
            meta.ns.for_all_neighbors(p_i, self.divergence_solver_iteration_task, ret)
            meta.pd.v[p_i] += ret.dv

    @ti.func
    def divergence_solver_iteration_task(self, p_i, p_j, ret: ti.template()):
        if meta.pd.material[p_j] == FLUID:
            # Fluid neighbors
            b_j = meta.pd.density_adv[p_j]
            k_j = b_j * meta.pd.dfsph_factor[p_j]
            k_sum = (
                ret.k_i + self.density_0 / self.density_0 * k_j
            )  # TODO: make the neighbor density0 different for multiphase fluid
            if ti.abs(k_sum) > self.m_eps:
                grad_p_j = -meta.pd.m_V[p_j] * self.grad_w_ij(p_i, p_j)
                ret.dv -= self.dt[None] * k_sum * grad_p_j
        elif meta.pd.material[p_j] == SOLID:
            # Boundary neighbors
            ## Akinci2012
            if ti.abs(ret.k_i) > self.m_eps:
                grad_p_j = -meta.pd.m_V[p_j] * self.grad_w_ij(p_i, p_j)
                vel_change = -self.dt[None] * 1.0 * ret.k_i * grad_p_j
                ret.dv += vel_change
                if is_dynamic_rigid_body(p_j):
                    meta.pd.acceleration[p_j] += (
                        # -vel_change * (1 / self.dt[None]) * meta.pd.density[p_i] / meta.pd.density[p_j]
                        -vel_change
                        * (1 / self.dt[None])
                    )

    def pressure_solve(self):
        inv_dt = 1 / self.dt[None]
        inv_dt2 = 1 / (self.dt[None] * self.dt[None])

        # TODO: warm start

        # Compute rho_adv
        self.compute_density_adv()

        self.multiply_time_step(meta.pd.dfsph_factor, inv_dt2)

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
        return density_err / meta.fluid_particle_num

    @ti.kernel
    def pressure_solve_iteration_kernel(self):
        # Compute pressure forces
        for p_i in ti.grouped(meta.pd.x):
            if meta.pd.material[p_i] != FLUID:
                continue
            # Evaluate rhs
            b_i = meta.pd.density_adv[p_i] - 1.0
            k_i = b_i * meta.pd.dfsph_factor[p_i]

            # TODO: if warmstart
            # get kappa V
            meta.ns.for_all_neighbors(p_i, self.pressure_solve_iteration_task, k_i)

    @ti.func
    def pressure_solve_iteration_task(self, p_i, p_j, k_i: ti.template()):
        if meta.pd.material[p_j] == FLUID:
            # Fluid neighbors
            b_j = meta.pd.density_adv[p_j] - 1.0
            k_j = b_j * meta.pd.dfsph_factor[p_j]
            k_sum = (
                k_i + self.density_0 / self.density_0 * k_j
            )  # TODO: make the neighbor density0 different for multiphase fluid
            if ti.abs(k_sum) > self.m_eps:
                grad_p_j = -meta.pd.m_V[p_j] * self.grad_w_ij(p_i, p_j)
                # Directly update velocities instead of storing pressure accelerations
                meta.pd.v[p_i] -= self.dt[None] * k_sum * grad_p_j  # ki, kj already contain inverse density
        elif meta.pd.material[p_j] == SOLID:
            # Boundary neighbors
            ## Akinci2012
            if ti.abs(k_i) > self.m_eps:
                grad_p_j = -meta.pd.m_V[p_j] * self.grad_w_ij(p_i, p_j)

                # Directly update velocities instead of storing pressure accelerations
                vel_change = -self.dt[None] * 1.0 * k_i * grad_p_j  # kj already contains inverse density
                meta.pd.v[p_i] += vel_change
                if is_dynamic_rigid_body(p_j):
                    meta.pd.acceleration[p_j] += (
                        # -vel_change * 1.0 / self.dt[None] * meta.pd.density[p_i] / meta.pd.density[p_j]
                        -vel_change
                        * 1.0
                        / self.dt[None]
                    )

    @ti.kernel
    def predict_velocity(self):
        # compute new velocities only considering non-pressure forces
        for p_i in ti.grouped(meta.pd.x):
            if meta.pd.is_dynamic[p_i] and meta.pd.material[p_i] == FLUID:
                meta.pd.v[p_i] += self.dt[None] * meta.pd.acceleration[p_i]

    def substep(self):
        self.compute_densities()
        print(f"max density: {meta.pd.density.to_numpy().max()}")
        self.compute_DFSPH_factor()
        if self.enable_divergence_solver:
            self.divergence_solve()
        self.compute_non_pressure_forces()
        self.predict_velocity()
        self.pressure_solve()


def make_domainbox():
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
    return box_anchors, box_lines_indices


def initialize():
    meta.parm = Parameter()
    meta.particle_max_num, meta.fluid_particle_num = load_particles()
    meta.pd = ParticleData(meta.particle_max_num)
    initialize_particles(meta.pd)  # fill the taichi fields

    meta.ns = NeighborhoodSearch()

    meta.rbs = []
    for phase in meta.phase_info.values():
        if phase.solid_type == RIGID:
            rb = RigidBody(meta.pd.x, phase.uid)
            # rb = RigidBodySolver(phase.uid)
            meta.rbs.append(rb)


# ---------------------------------------------------------------------------- #
#                                     main                                     #
# ---------------------------------------------------------------------------- #
def main():
    parser = argparse.ArgumentParser(description="SPH Taichi")
    substeps = get_cfg("numberOfStepsPerRenderUpdate")

    initialize()

    solver = build_solver()
    solver.initialize()

    if get_cfg("noGUI"):
        for _ in range(get_cfg("maxStep")):
            solver.step()
        return

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
    meta.frame = 0
    while window.running:
        for e in window.get_events(ti.ui.PRESS):
            if e.key == ti.ui.SPACE:
                meta.paused = not meta.paused
                print("paused:", meta.paused)
            if e.key == "f":
                print("Step once, step: ", meta.step_num)
                solver.step()
                meta.step_num += 1
        if not meta.paused:
            solver.step()
            meta.step_num += 1
            meta.frame += 1

        # print(camera.curr_position)
        # print(camera.curr_lookat)
        camera.track_user_inputs(window, movement_speed=movement_speed, hold_key=ti.ui.RMB)
        scene.set_camera(camera)
        scene.point_light((2.0, 2.0, 2.0), color=(1.0, 1.0, 1.0))
        # scene.particles(meta.pd.x, radius=meta.parm.particle_radius, color=WHITE)
        scene.particles(meta.pd.x, radius=meta.parm.particle_radius, per_vertex_color=meta.pd.color)
        scene.lines(box_anchors, indices=box_lines_indices, color=(0.99, 0.68, 0.28), width=1.0)
        canvas.scene(scene)
        cnt += 1
        window.show()


if __name__ == "__main__":
    main()
