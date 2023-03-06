from luisa import *
from luisa.types import *
from luisa.builtin import *
import numpy as np
import random as rand
import math
init("dx")
n_grid = 32
n_steps = 25

n_particles = n_grid * n_grid * n_grid // 4
dx = 1. / n_grid
dt = 8e-5
p_rho = 1.
p_vol = (dx * .5) * (dx * .5) * (dx * .5)
p_mass = p_rho * p_vol
gravity = 9.8
bound = 3
E = 400.
resolution = 1024

Float3Array3 = ArrayType(3, float3)
x = Buffer(n_particles, float3)
v = Buffer(n_particles, float3)
C = Buffer(n_particles, float3x3)
J = Buffer(n_particles, float)
grid_v = Buffer(n_grid * n_grid * n_grid * 4, float)
grid_m = Buffer(n_grid * n_grid * n_grid, float)
display = Texture2D(resolution, resolution, 4, float, storage="BYTE")


@func
def index(xyz: int3):
    p = clamp(xyz, int3(0), int3(n_grid - 1))
    return p.x + p.y * n_grid + p.z * n_grid * n_grid


@func
def outer_product(a, b):
    return float3x3(
        float3(a[0] * b[0], a[1] * b[0], a[2] * b[0]),
        float3(a[0] * b[1], a[1] * b[1], a[2] * b[1]),
        float3(a[0] * b[2], a[1] * b[2], a[2] * b[2]))


@func
def trace(m):
    return m[0][0] + m[1][1] + m[2][2]


@func
def clear_grid():
    set_block_size(16, 16, 1)
    idx = index(dispatch_id())
    grid_v.write(idx * 4, 0.)
    grid_v.write(idx * 4 + 1, 0.)
    grid_v.write(idx * 4 + 2, 0.)
    grid_v.write(idx * 4 + 3, 0.)
    grid_m.write(idx, 0.)


@func
def sqr(x):
    return x * x


@func
def point_to_grid():
    set_block_size(256, 1, 1)
    p = dispatch_id().x
    Xp = x.read(p) / dx
    base = int3(Xp - 0.5)
    fx = Xp - float3(base)
    w = Float3Array3()
    w[0] = float3(0.5) * sqr(float3(1.5) - fx)
    w[1] = float3(0.75) - sqr(fx - float3(1.0))
    w[2] = float3(0.5) * sqr(fx - float3(0.5))
    stress = -4. * dt * E * p_vol * (J.read(p) - 1.) / sqr(dx)
    affine = float3x3(stress) + p_mass * C.read(p)
    vp = v.read(p)
    for ii in range(27):
        offset = int3(ii % 3, (ii // 3) % 3, ii // 3 // 3)
        i = offset.x
        j = offset.y
        k = offset.z
        dpos = (float3(offset) - fx) * dx
        weight = w[i].x * w[j].y * w[k].z
        vadd = weight * (p_mass * vp + affine * dpos)
        idx = index(base + offset)
        old = grid_v.atomic_fetch_add(idx * 4, vadd.x)
        old = grid_v.atomic_fetch_add(idx * 4 + 1, vadd.y)
        old = grid_v.atomic_fetch_add(idx * 4 + 2, vadd.z)
        old = grid_m.atomic_fetch_add(idx, weight * p_mass)


@func
def ite(a, b, c):
    return select(c, b, a)


@func
def simulate_grid():
    set_block_size(16, 16, 1)
    coord = dispatch_id().xyz
    i = index(coord)
    v = float3(grid_v.read(i * 4), grid_v.read(i * 4 + 1),
               grid_v.read(i * 4 + 2))
    m = grid_m.read(i)
    v = ite(m > float3(0.), v / m, v)
    v.y -= dt * gravity
    v = ite((coord < bound and v < float3(0.)) or (
        coord > n_grid - bound and v > float3(0.)), float3(0.), v)
    grid_v.write(i * 4, v.x)
    grid_v.write(i * 4 + 1, v.y)
    grid_v.write(i * 4 + 2, v.z)
    grid_v.write(i * 4 + 3, 0.)


@func
def grid_to_point():
    set_block_size(256, 1, 1)
    p = dispatch_id().x
    Xp = x.read(p) / dx
    base = int3(Xp - 0.5)
    fx = Xp - float3(base)
    w = Float3Array3()
    w[0] = float3(0.5) * sqr(float3(1.5) - fx)
    w[1] = float3(0.75) - sqr(fx - float3(1.0))
    w[2] = float3(0.5) * sqr(fx - float3(0.5))
    new_v = float3(0)
    new_C = float3x3(0)
    for ii in range(27):
        offset = int3(ii % 3, (ii // 3) % 3, ii // 3 // 3)
        i = offset.x
        j = offset.y
        k = offset.z
        dpos = (float3(offset) - fx) * dx
        weight = w[i].x * w[j].y * w[k].z
        idx = index(base + offset)
        g_v = float3(grid_v.read(idx * 4),
                     grid_v.read(idx * 4 + 1),
                     grid_v.read(idx * 4 + 2))
        new_v += weight * g_v
        new_C = new_C + 4. * weight * outer_product(g_v, dpos) / sqr(dx)
    v.write(p, new_v)
    x.write(p, x.read(p) + new_v * dt)
    J.write(p, J.read(p) * (1. + dt * trace(new_C)))
    C.write(p, new_C)


@func
def clear_display():
    set_block_size(16, 16, 1)
    display.write(dispatch_id().xy, float4(.2, .2, .2, 1.))


def radians(deg):
    return deg * math.pi / 180.


phi = radians(28.)
theta = radians(32.)


@func
def T(a0: float3):
    a = a0 - 0.5
    c = cos(phi)
    s = sin(phi)
    C = cos(theta)
    S = sin(theta)
    a.x = a.x * c + a.z * s
    a.z = a.z * c - a.x * s
    return float2(a.x, a.y * C + a.z * S) + 0.5


@func
def draw_particles():
    set_block_size(256, 1, 1)
    p = dispatch_id().x
    basepos = T(x.read(p))
    for i in range(-1, 2):
        for j in range(-1, 2):
            pos = int2(basepos * float(resolution)) + int2(i, j)
            if (pos.x >= 0 and pos.x < resolution and pos.y >= 0 and pos.y < resolution):
                display.write(uint2(pos.x, resolution - 1 -
                              pos.y), float4(1., 1., 1., 1.))


def substep():
    clear_grid(dispatch_size=(n_grid, n_grid, n_grid))
    point_to_grid(dispatch_size=(n_particles, 1, 1))
    simulate_grid(dispatch_size=(n_grid, n_grid, n_grid))
    grid_to_point(dispatch_size=(n_particles, 1, 1))


def init_value():
    x_init = np.empty(n_particles * 4, dtype=np.float32)
    for i in range(n_particles):
        rx = rand.random()
        ry = rand.random()
        rz = rand.random()
        x_init[i * 4] = rx * .4 + .2
        x_init[i * 4 + 1] = ry * .4 + .2
        x_init[i * 4 + 2] = rz * .4 + .2
    v_init = np.zeros(n_particles * 4, dtype=np.float32)
    J_init = np.ones(n_particles, dtype=np.float32)
    C_init = np.zeros(n_particles * 12, dtype=np.float32)
    x.copy_from(x_init)
    v.copy_from(v_init)
    J.copy_from(J_init)
    C.copy_from(C_init)
    synchronize()


init_value()
gui = GUI("Test MPM", (resolution, resolution))
while gui.running():
    for i in range(n_steps):
        substep()
    clear_display(dispatch_size=(resolution, resolution, 1))
    draw_particles(dispatch_size=(n_particles, 1, 1))
    execute()
    gui.set_image(display)
    gui.show()
synchronize()
