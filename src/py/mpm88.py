# MPM-MLS in 88 lines of Taichi code, originally created by @yuanming-hu
# ported to Luisa

import luisa as lc
from luisa.mathtypes import *

lc.init()

n_particles = 8192
n_grid = 128
dx = 1 / n_grid
dt = 2e-4

p_rho = 1
p_vol = (dx * 0.5)**2
p_mass = p_vol * p_rho
gravity = 9.8
bound = 3
E = 400

x = lc.Buffer.empty(n_particles, dtype=float2)
v = lc.Buffer.empty(n_particles, dtype=float2)
C = lc.Buffer.zeros(n_particles, dtype=float2x2)
J = lc.Buffer.empty(n_particles, dtype=float)

grid_v = lc.Texture2D(n_grid, n_grid, 2, dtype=float)
grid_m = lc.Texture2D(n_grid, n_grid, 1, dtype=float)
grid_mutex = lc.Buffer.zeros(n_grid * n_grid, dtype=int)


@lc.func
def clear_grid():
    grid_v.write(dispatch_id().xy, float2(0))
    grid_m.write(dispatch_id().xy, 0.)

@lc.func
def point_to_grid():
    p = dispatch_id().x
    Xp = x.read(p) / dx
    base = int2(Xp - 0.5)
    fx = Xp - float2(base)
    if base.x < 0 or base.y < 0 or base.x+2 >= n_grid or base.y+2 >= n_grid:
        print("base:", base)
        print("[ABORT]")
    w = array([0.5 * (1.5 - fx)**2, 0.75 - (fx - 1.0)**2, 0.5 * (fx - 0.5)**2])
    stress = -dt * 4 * E * p_vol * (J.read(p) - 1) / dx**2
    affine = float2x2(stress) + p_mass * C.read(p)
    for i in range(3):
        for j in range(3):
            offset = int2(i,j)

            tmp = base + offset
            if tmp.x < 0 or tmp.y < 0 or tmp.x >= n_grid or tmp.y >= n_grid:
                print("WTF")
                print("[ABORT]")
                return

            dpos = (float2(offset) - fx) * dx
            weight = w[i].x * w[j].y
            new_v = grid_v.read(base + offset) + weight * (p_mass * v.read(p) + affine * dpos)
            if abs(new_v.x) > 1000 or abs(new_v.y) > 1000:
                print("!!!! new_v", new_v)
                print("[ABORT]")
            grid_mutex.lock
            grid_v.write(base + offset, grid_v.read(base + offset) + weight * (p_mass * v.read(p) + affine * dpos))
            grid_m.write(base + offset, grid_m.read(base + offset) + weight * p_mass)
            grid_mutex.unlock

@lc.func
def simulate_grid():
    coord = dispatch_id().xy
    v = grid_v.read(coord)
    m = grid_m.read(coord)
    if m > 0.0:
        v /= m
    v.y -= dt * gravity
    if coord.x < bound and v.x < 0.0:
        v.x = 0.0
    if coord.x > n_grid - bound and v.x > 0.0:
        v.x = 0.0
    if coord.y < bound and v.y < 0.0:
        v.y = 0.0
    if coord.y > n_grid - bound and v.y > 0.0:
        v.y = 0.0

    if abs(v.x) > 1000 or abs(v.y) > 1000:
        print("!!!! v", v, "m", m)
        print("[ABORT]")
    grid_v.write(coord, v)

@lc.func
def outer_product(a: float2, b: float2):
    return float2x2(a[0]*b[0], a[1]*b[0], a[0]*b[1], a[1]*b[1])

@lc.func
def trace(a: float2x2):
    return a[0][0] + a[1][1]

@lc.func
def grid_to_point():
    p = dispatch_id().x
    Xp = x.read(p) / dx
    base = int2(Xp - 0.5)
    if base.x < 0 or base.y < 0 or base.x >= n_grid or base.y >= n_grid:
        print("base:", base)
    fx = Xp - float2(base)
    w = array([0.5 * (1.5 - fx)**2, 0.75 - (fx - 1.0)**2, 0.5 * (fx - 0.5)**2])
    new_v = float2(0)
    new_C = float2x2(0)
    for i in range(3):
        for j in range(3):
            offset = int2(i,j)
            dpos = (float2(offset) - fx) * dx
            weight = w[i].x * w[j].y
            g_v = grid_v.read(base + offset)
            new_v += weight * g_v
            new_C += 4 * weight * outer_product(g_v, dpos) / dx**2
    v.write(p, new_v)
    new_x = x.read(p) + dt * new_v
    if (new_x.x < 0 or new_x.y < 0) and x.read(p).x >= 0 and x.read(p).y >= 0:
        print(f"escaping No.{p}: {x.read(p)} -> {new_x} @v={new_v}")
        print("w=", w)
        for i in range(3):
            for j in range(3):
                offset = int2(i,j)
                g_v = grid_v.read(base + offset)
                print(f"g_v[{base+offset}] = {g_v}; weight={w[i].x * w[j].y}")
        print("[ABORT]")
    x.write(p, x.read(p) + dt * new_v)
    # if (x.read(p).y < 0):
    #     print("!!!!", x.read(p), new_v)
    J.write(p, J.read(p) * (1 + dt * trace(new_C)))
    C.write(p, new_C)


def substep():
    print("clear grid")
    clear_grid(dispatch_size=(n_grid, n_grid))
    lc.synchronize()
    print("point_to_grid")
    point_to_grid(dispatch_size=n_particles)
    lc.synchronize()
    print("simulate_grid")
    simulate_grid(dispatch_size=(n_grid, n_grid))
    lc.synchronize()
    print("grid_to_point")
    grid_to_point(dispatch_size=n_particles)
    lc.synchronize()


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
            pos = int2(x.read(p) * float(res)) + int2(i,j)
            if pos.x >= 0 and pos.y >= 0 and pos.x < res and pos.y < res:
                display.write(pos, float4(0.4, 0.6, 0.6, 1.0))

init(dispatch_size=n_particles)
lc.synchronize()
for i in range(10000):
    print("==============", i, "=============")
    substep()
quit()

gui = lc.GUI('MPM88', (res,res))
while gui.running():
    for s in range(5):
        substep()
    clear_display(dispatch_size=(res,res))
    draw_particle(dispatch_size=n_particles)
    gui.set_image(display)
    gui.show()
