# MPM-MLS in 88 lines of Taichi code, originally created by @yuanming-hu
# ported to Luisa

from sys import argv
import luisa as lc
from luisa.mathtypes import *

if len(argv) > 1:
    lc.init(argv[1])
else:
    lc.init()

n_grid = 128
n_steps = 16
n_particles = n_grid ** 2 // 2
dx = 1 / n_grid
dt = 1e-4

p_rho = 1
p_vol = (dx * 0.5) ** 2
p_mass = p_vol * p_rho
gravity = 9.8
bound = 3
E = 400

x = lc.Buffer.empty(n_particles, dtype=float2)
v = lc.Buffer.empty(n_particles, dtype=float2)
C = lc.Buffer.zeros(n_particles, dtype=float2x2)
J = lc.Buffer.empty(n_particles, dtype=float)

grid_v = lc.Buffer.empty(n_grid * n_grid * 2, dtype=float)
grid_m = lc.Buffer.empty(n_grid * n_grid, dtype=float)


@lc.func
def encode(pos: int2):
    p = clamp(pos, int2(0), int2(n_grid - 1))
    return p.x + p.y * n_grid


@lc.func
def clear_grid():
    set_block_size(16, 16, 1)
    grid_v.write(encode(dispatch_id().xy) * 2, 0.)
    grid_v.write(encode(dispatch_id().xy) * 2 + 1, 0.)
    grid_m.write(encode(dispatch_id().xy), 0.)


@lc.func
def point_to_grid():
    set_block_size(64, 1, 1)
    p = dispatch_id().x
    Xp = x.read(p) / dx
    base = int2(Xp - 0.5)
    fx = Xp - float2(base)
    w = array([0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1.0) ** 2, 0.5 * (fx - 0.5) ** 2])
    stress = -dt * 4 * E * p_vol * (J.read(p) - 1) / dx ** 2
    affine = float2x2(stress) + p_mass * C.read(p)
    vp = v.read(p)
    for ii in range(9):
        offset = int2(ii % 3, ii // 3)
        i = offset.x
        j = offset.y
        dpos = (float2(offset) - fx) * dx
        weight = w[i].x * w[j].y
        vadd = weight * (p_mass * vp + affine * dpos)
        idx = encode(base + offset)
        _ = grid_v.atomic_fetch_add(idx * 2, vadd.x)
        _ = grid_v.atomic_fetch_add(idx * 2 + 1, vadd.y)
        _ = grid_m.atomic_fetch_add(idx, weight * p_mass)


@lc.func
def simulate_grid():
    set_block_size(16, 16, 1)
    coord = dispatch_id().xy
    i = encode(coord)
    v = float2(grid_v.read(i * 2), grid_v.read(i * 2 + 1))
    m = grid_m.read(i)
    v = v / m if m > 0.0 else v
    v.y -= dt * gravity
    v.x = 0.0 if coord.x < bound and v.x < 0.0 or coord.x > n_grid - bound and v.x > 0.0 else v.x
    v.y = 0.0 if coord.y < bound and v.y < 0.0 or coord.y > n_grid - bound and v.y > 0.0 else v.y
    grid_v.write(i * 2, v.x)
    grid_v.write(i * 2 + 1, v.y)


@lc.func
def outer_product(a: float2, b: float2):
    return float2x2(a[0] * b[0], a[1] * b[0], a[0] * b[1], a[1] * b[1])


@lc.func
def trace(a: float2x2):
    return a[0][0] + a[1][1]


@lc.func
def grid_to_point():
    set_block_size(64, 1, 1)
    p = dispatch_id().x
    Xp = x.read(p) / dx
    base = int2(Xp - 0.5)
    fx = Xp - float2(base)
    w = array([0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1.0) ** 2, 0.5 * (fx - 0.5) ** 2])
    new_v = float2(0)
    new_C = float2x2(0)
    for ii in range(9):
        offset = int2(ii % 3, ii // 3)
        i = offset.x
        j = offset.y
        dpos = (float2(offset) - fx) * dx
        weight = w[i].x * w[j].y
        idx = encode(base + offset)
        g_v = float2(grid_v.read(idx * 2), grid_v.read(idx * 2 + 1))
        new_v += weight * g_v
        new_C += 4 * weight * outer_product(g_v, dpos) / dx ** 2
    v.write(p, new_v)
    x.write(p, x.read(p) + dt * new_v)
    J.write(p, J.read(p) * (1 + dt * trace(new_C)))
    C.write(p, new_C)


def substep():
    clear_grid(dispatch_size=(n_grid, n_grid))
    point_to_grid(dispatch_size=n_particles)
    simulate_grid(dispatch_size=(n_grid, n_grid))
    grid_to_point(dispatch_size=n_particles)


@lc.func
def init():
    sampler = lc.RandomSampler(dispatch_id())
    i = dispatch_id().x
    x.write(i, sampler.next2f() * 0.4 + 0.2)
    v.write(i, float2(0, -1))
    J.write(i, 1.0)


res = 512
display = lc.Texture2D(res, res, 4, dtype=float)


@lc.func
def clear_display():
    coord = dispatch_id().xy
    display.write(coord, float4(0.1, 0.2, 0.3, 1.0))


@lc.func
def draw_particle():
    p = dispatch_id().x
    for i in range(-1, 2):
        for j in range(-1, 2):
            pos = int2(x.read(p) * float(res)) + int2(i, j)
            if pos.x >= 0 and pos.y >= 0 and pos.x < res and pos.y < res:
                display.write(make_int2(pos.x, res - 1 - pos.y), float4(0.4, 0.6, 0.6, 1.0))


init(dispatch_size=n_particles)

gui = lc.GUI('MPM88', (res, res))
while gui.running():
    for s in range(n_steps):
        substep()
    clear_display(dispatch_size=(res, res))
    draw_particle(dispatch_size=n_particles)
    gui.set_image(display)
    gui.show()
