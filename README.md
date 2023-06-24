本repo是实现 An Unified Particle-Based Framework for non-Newtonian Behaviours Simulation 的代码

## 1. File Structure

SPH: 本论文的原本的代码。
- data/scenes: 场景文件
- data/models: 模型文件
- main.py: 所有代码均放在这一个文件中

MPM: 要对比的MPM代码。采用Su21(A Unified Second-Order Accurate in Time MPM Formulation for Simulating Viscoelastic Liquids with Phase Change, https://orionquest.github.io/papers/USOSVLPC/paper.html) 的算法。基于其MATLAB代码改写。

## 2. Roadmap
Roadmap
- rigid body one way coupling
- multi-phase
- diffusion
- phase change control
- weiler viscosity
- non-Newtonian viscosity
- Becker elasticity
- plasticity and viscoelasticity

## 3. Code Convention
meta 是一个全局object, 所有全局变量都挂在meta上面。好处是可以在任何地方访问到这些全局变量。但一定要注意执行顺序。

- meta.pd: ParticleData object 所有粒子数据变量, 均为 taichi field。例如:
  - meta.pd.x 是粒子位置,
  - meta.pd.v 是粒子速度
  - meta.pd.acceleration是粒子加速度等等。
- meta.parm: 所有constant参数
- meta.ns: NeighborhoodSearch。
