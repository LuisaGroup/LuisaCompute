import math
import time
from unittest import result

import numpy as np

import taichi as ti
import luisa
from luisa.mathtypes import *

luisa.init("cuda")
res = 1280, 720
color_buffer = luisa.Buffer(1280 * 720 * 3, float)
max_ray_depth = 6
eps = 1e-4
inf = 1e10

fov = 0.23
dist_limit = 100

camera_pos = float3(0.0, 0.32, 3.7)
light_pos = float3(-1.5, 0.6, 0.3)
light_normal = float3(1.0, 0.0, 0.0)
light_radius = 2.0

next_hit_struct = luisa.StructType(cloest=float, normal=float3, c=float3)


@luisa.callable
def intersect_light(pos: float3, d: float3):
    vdot = luisa.dot(-d, light_normal)
    dist = luisa.dot(d, light_pos - pos)
    dist_to_light = inf
    if vdot > 0 and dist > 0:
        D = dist / vdot
        dist_to_center = luisa.distance_squared(light_pos, pos + D * d)
        if dist_to_center < light_radius**2:
            dist_to_light = D
    return dist_to_light


@luisa.callable
def out_dir(n: float3):
    u = float3(1.0, 0.0, 0.0)
    if abs(n.y) < 1 - eps:
        u = luisa.normalize(luisa.cross(n, float3(0.0, 1.0, 0.0)))
    v = luisa.cross(n, u)
    phi = 2 * math.pi * ti.random()
    ay = luisa.sqrt(ti.random())
    ax = luisa.sqrt(1 - ay ** 2)
    return ax * (luisa.cos(phi) * u + luisa.sin(phi) * v) + ay * n


@luisa.callable
def make_nested(f: float):
    f = f * 40
    i = int(f)
    if f < 0:
        if i % 2 == 1:
            f -= luisa.floor(f)
        else:
            f = luisa.floor(f) + 1 - f
    f = (f - 0.2) / 40
    return f


# https://www.iquilezles.org/www/articles/distfunctions/distfunctions.htm
@luisa.callable
def sdf(o):
    wall = luisa.min(o[1] + 0.1, o[2] + 0.4)
    sphere = luisa.length(o - float3(0.0, 0.35, 0.0)) - 0.36

    q = abs(o - float3(0.8, 0.3, 0)) - float3(0.3, 0.3, 0.3)
    box = luisa.length(luisa.max(q, 0)) + luisa.min(luisa.max(q.x, luisa.max(q.y, q.z)), 0)

    O = o - float3(-0.8, 0.3, 0)
    d = float2(luisa.length(float2(O.x, O.z)) - 0.3, luisa.abs(O.y) - 0.3)
    cylinder = min(luisa.max(d.x, d.y), 0.0) + luisa.length(luisa.max(d, 0))

    geometry = make_nested(luisa.min(luisa.min(sphere, box), cylinder))
    geometry = luisa.max(geometry, -(0.32 - (o.y * 0.6 + o.z * 0.8)))
    return luisa.min(wall, geometry)


@luisa.callable
def ray_march(p, d):
    j = 0
    dist = 0.0
    while j < 100 and sdf(p + dist * d) > 1e-6 and dist < inf:
        dist += sdf(p + dist * d)
        j += 1
    return luisa.min(inf, dist)


@luisa.callable
def sdf_normal(p):
    d = 1e-3
    n = float3(0.0, 0.0, 0.0)
    sdf_center = sdf(p)
    for i in range(3):
        inc = p
        inc[i] += d
        n[i] = (1 / d) * (sdf(inc) - sdf_center)
    return luisa.normalize(n)


@luisa.callable
def next_hit(pos, d):
    closest, normal, c = inf, float3(0, 0, 0), float3(0, 0, 0)
    ray_march_dist = ray_march(pos, d)
    if ray_march_dist < dist_limit and ray_march_dist < closest:
        closest = ray_march_dist
        normal = sdf_normal(pos + d * closest)
        hit_pos = pos + d * closest
        t = int((hit_pos[0] + 10) * 1.1 + 0.5) % 3
        c = float3(
            0.4 + 0.3 * (t == 0), 0.4 + 0.2 * (t == 1), 0.4 + 0.3 * (t == 2))
    result = next_hit_struct()
    result.closest = closest
    result.normal = normal
    result.c = c
    return result


@luisa.kernel
def render():
    for u, v in color_buffer:
        aspect_ratio = res[0] / res[1]
        pos = camera_pos
        d = float2(
            (2 * fov * (u + ti.random()) / res[1] - fov * aspect_ratio - 1e-5),
            2 * fov * (v + ti.random()) / res[1] - fov - 1e-5, -1.0
        )
        d = luisa.normalize(d)

        throughput = float3(1.0, 1.0, 1.0)

        depth = 0
        hit_light = 0.00

        while depth < max_ray_depth:
            result = next_hit(pos, d)
            closest = result.closest
            normal = result.normal
            c = result.c
            depth += 1
            dist_to_light = intersect_light(pos, d)
            if dist_to_light < closest:
                hit_light = 1
                depth = max_ray_depth
            else:
                hit_pos = pos + closest * d
                if luisa.distance_squared(0, normal) != 0:
                    d = out_dir(normal)
                    pos = hit_pos + 1e-4 * d
                    throughput *= c
                else:
                    depth = max_ray_depth
        adder = throughput * hit_light
        color_buffer[u * res[1] * 3 + v * 3 + 0] += adder.x
        color_buffer[u * res[1] * 3 + v * 3 + 1] += adder.y
        color_buffer[u * res[1] * 3 + v * 3 + 2] += adder.z


gui = ti.GUI('SDF Path Tracer', res)
last_t = 0
for i in range(50000):
    render()
    interval = 10
    if i % interval == 0 and i > 0:
        print("{:.2f} samples/s".format(interval / (time.time() - last_t)))
        last_t = time.time()
        img = color_buffer.to_numpy() * (1 / (i + 1))
        img = img / img.mean() * 0.24
        gui.set_image(np.sqrt(img))
        gui.show()