# mpm3d ported to Luisa

import numpy as np
import luisa as lc
from luisa.mathtypes import *
import math
from os import makedirs

lc.init("metal")

# dim, n_grid, steps, dt = 2, 128, 20, 2e-4
# dim, n_grid, steps, dt = 2, 256, 32, 1e-4
# dim, n_grid, steps, dt = 3, 32, 25, 4e-4
dim, n_grid, steps, dt = 3, 64, 25, 2e-4
# dim, n_grid, steps, dt = 3, 128, 25, 8e-5

n_particles = n_grid ** dim // 2 ** (dim - 1)
dx = 1 / n_grid

p_rho = 1
p_vol = (dx * 0.5) ** 2
p_mass = p_vol * p_rho
gravity = 9.8
bound = 3
E = 400

x = lc.Buffer.empty(n_particles, dtype=float3)
v = lc.Buffer.empty(n_particles, dtype=float3)
C = lc.Buffer.empty(n_particles, dtype=float3x3)
J = lc.Buffer.empty(n_particles, dtype=float)

grid_v = lc.Buffer.empty(n_grid ** dim * 4, dtype=float)
grid_m = lc.Buffer.empty(n_grid ** dim, dtype=float)

neighbour = lc.array([int3(i, j, k) for i in range(3) for j in range(3) for k in range(3)])


@lc.func
def encode(pos: int3):
    return pos.x + pos.y * n_grid + pos.z * n_grid * n_grid


@lc.func
def clear_grid():
    grid_v.write(encode(dispatch_id()) * 4 + 0, 0.)
    grid_v.write(encode(dispatch_id()) * 4 + 1, 0.)
    grid_v.write(encode(dispatch_id()) * 4 + 2, 0.)
    # grid_v.write(encode(dispatch_id())*4+3, 0.)
    grid_m.write(encode(dispatch_id()), 0.)


# ti.block_dim(n_grid)


@lc.func
def point_to_grid():
    p = dispatch_id().x
    Xp = x.read(p) / dx
    base = int3(Xp - 0.5)
    fx = Xp - float3(base)
    w = array([0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1.0) ** 2, 0.5 * (fx - 0.5) ** 2])
    stress = -dt * 4 * E * p_vol * (J.read(p) - 1) / dx ** 2
    affine = float3x3(stress) + p_mass * C.read(p)
    for offset in neighbour:
        dpos = (float3(offset) - fx) * dx
        weight = w[offset[0]][0] * w[offset[1]][1] * w[offset[2]][2]
        vadd = weight * (p_mass * v.read(p) + affine * dpos)
        _ = grid_v.atomic_fetch_add(encode(base + offset) * 4 + 0, vadd[0])
        _ = grid_v.atomic_fetch_add(encode(base + offset) * 4 + 1, vadd[1])
        _ = grid_v.atomic_fetch_add(encode(base + offset) * 4 + 2, vadd[2])
        _ = grid_m.atomic_fetch_add(encode(base + offset), weight * p_mass)


@lc.func
def simulate_grid():
    I = dispatch_id()
    v = float3(grid_v.read(encode(I) * 4 + 0),
               grid_v.read(encode(I) * 4 + 1),
               grid_v.read(encode(I) * 4 + 2))
    m = grid_m.read(encode(I))
    if m > 0.0:
        v /= m
    v.y -= dt * gravity
    cond = I < bound and v < 0. or I > n_grid - bound and v > 0.
    v = float3(0) if cond else v
    grid_v.write(encode(I) * 4 + 0, v[0])
    grid_v.write(encode(I) * 4 + 1, v[1])
    grid_v.write(encode(I) * 4 + 2, v[2])


# # ti.block_dim(n_grid)

@lc.func
def outer_product(a: float3, b: float3):
    return float3x3(
        float3(a[0] * b[0], a[1] * b[0], a[2] * b[0]),
        float3(a[0] * b[1], a[1] * b[1], a[2] * b[1]),
        float3(a[0] * b[2], a[1] * b[2], a[2] * b[2]))


@lc.func
def trace(a: float3x3):
    return a[0][0] + a[1][1] + a[2][2]


@lc.func
def grid_to_point():
    p = dispatch_id().x
    Xp = x.read(p) / dx
    base = int3(Xp - 0.5)
    fx = Xp - float3(base)
    w = array([0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1.0) ** 2, 0.5 * (fx - 0.5) ** 2])
    new_v = float3(0)
    new_C = float3x3(0)
    for offset in neighbour:
        dpos = (float3(offset) - fx) * dx
        weight = w[offset[0]][0] * w[offset[1]][1] * w[offset[2]][2]
        g_v = float3(grid_v.read(encode(base + offset) * 4 + 0),
                     grid_v.read(encode(base + offset) * 4 + 1),
                     grid_v.read(encode(base + offset) * 4 + 2))
        new_v += weight * g_v
        new_C += 4 * weight * outer_product(g_v, dpos) / dx ** 2

    v.write(p, new_v)
    x.write(p, x.read(p) + dt * new_v)
    J.write(p, J.read(p) * (1 + dt * trace(new_C)))
    C.write(p, new_C)


def substep():
    clear_grid(dispatch_size=(n_grid,) * 3)
    point_to_grid(dispatch_size=n_particles)
    simulate_grid(dispatch_size=(n_grid,) * 3)
    grid_to_point(dispatch_size=n_particles)


@lc.func
def init():
    sampler = lc.RandomSampler(dispatch_id())
    i = dispatch_id().x
    x.write(i, sampler.next3f() * 0.4 + 0.15)
    v.write(i, float3(0))
    C.write(i, float3x3(0))
    J.write(i, 1.0)


phi = math.radians(28)
theta = math.radians(32)


@lc.func
def T(a0: float3):
    a = a0 - 0.5
    c = cos(phi)
    s = sin(phi)
    C = cos(theta)
    S = sin(theta)
    a.x, a.z = a.x * c + a.z * s, a.z * c - a.x * s
    u, v = a.x, a.y * C + a.z * S
    return float2(u, v) + 0.5


res = 512
display = lc.Texture2D(res, res, 4, dtype=float)


@lc.func
def clear_display():
    coord = dispatch_id().xy
    display.write(coord, float4(0.1, 0.2, 0.3, 1.0))


@lc.func
def draw_particle():
    p = dispatch_id().x
    basepos = T(x.read(p))
    for i in range(-2, 3):
        for j in range(-2, 3):
            pos = int2(basepos * float(res)) + int2(i, j)
            coord = make_int2(pos.x, res - 1 - pos.y)
            if all(coord >= 0 and coord < res):
                old = display.read(coord)
                t = (24000 / n_particles) / ((i * i + j * j) + 1)
                display.write(coord, float4(lerp(old.xyz, 1.0, t), 1.0))


init(dispatch_size=n_particles)
points = np.zeros(shape=[n_particles, 4], dtype=np.float32)
out_folder = "mpm3d_outputs"
makedirs(out_folder, exist_ok=True)
gui = lc.GUI('MPM88', (res, res))
frame_id = 0
while gui.running():
    for s in range(steps):
        substep()
    print(f"Saving Frame #{frame_id}")
    x.copy_to(points)
    np.savetxt(f"{out_folder}/{frame_id:05}.txt", points[:, :3])
    frame_id += 1
    clear_display(dispatch_size=(res, res))
    draw_particle(dispatch_size=n_particles)
    gui.set_image(display)
    gui.show()
