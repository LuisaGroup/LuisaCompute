import math
from time import perf_counter
import luisa
from luisa.mathtypes import *
import numpy as np
from sys import argv


if len(argv) > 1:
    luisa.init(argv[1])
else:
    luisa.init()
res = 1280, 720
image = luisa.Texture2D.empty(*res, 4, float)
max_ray_depth = 6
eps = 1e-4
inf = 1e10

fov = 0.23
dist_limit = 100

camera_pos = float3(0.0, 0.32, 3.7)
light_pos = float3(-1.5, 0.6, 0.3)
light_normal = float3(1.0, 0.0, 0.0)
light_radius = 2.0

next_hit_struct = luisa.StructType(closest=float, normal=float3, c=float3)


@luisa.func
def intersect_light(pos, d):
    cosl = dot(-d, light_normal)
    dist = dot(d, light_pos - pos)
    dist_to_light = inf
    if cosl > 0 and dist > 0:
        D = dist / cosl
        dist_to_center = length_squared(light_pos - pos - D * d)
        if dist_to_center < light_radius ** 2:
            dist_to_light = D
    return dist_to_light

@luisa.func
def out_dir(n, sampler):
    u = float3(1.0, 0.0, 0.0)
    if abs(n.y) < 1 - eps:
        u = normalize(cross(n, float3(0,1,0)))
    v = cross(n, u)
    phi = 2 * math.pi * sampler.next()
    ay = sqrt(sampler.next())
    ax = sqrt(1 - ay ** 2)
    return ax * (cos(phi) * u + sin(phi) * v) + ay * n


@luisa.func
def make_nested(f):
    f = f * 40.0
    i = int(f)
    if f < 0:
        if i % 2 != 0:
            f -= floor(f)
        else:
            f = floor(f) + 1 - f
    f = (f - 0.2) / 40
    return f

# https://www.iquilezles.org/www/articles/distfunctions/distfunctions.htm
@luisa.func
def sdf(o):
    wall = min(o.y + 0.1, o.z + 0.4)
    sphere = length(o - float3(0.0, 0.35, 0.0)) - 0.36

    q = abs(o - float3(0.8, 0.3, 0.0)) - float3(0.3, 0.3, 0.3)
    box = length(max(q, 0.0)) + min(max(q.x, max(q.y, q.z)), 0.0)

    O = o - float3(-0.8, 0.3, 0.0)
    d = float2(length(float2(O.x, O.z)) - 0.3, abs(O.y) - 0.3)
    cylinder = min(max(d.x, d.y), 0.0) + length(max(d, 0.0))

    geometry = make_nested(min(min(sphere, box), cylinder))
    geometry = max(geometry, -(0.32 - (o.y * 0.6 + o.z * 0.8)))
    return min(wall, geometry)


@luisa.func
def ray_march(p, d):
    dist = 0.0
    for j in range(100):
        s = sdf(p + dist * d)
        if s <= 1e-6 or dist >= inf:
            break
        dist += s
    return min(inf, dist)


@luisa.func
def sdf_normal(p):
    d = 1e-3
    n = float3(0)
    sdf_center = sdf(p)
    for i in range(3):
        inc = p
        inc[i] += d
        n[i] = (1 / d) * (sdf(inc) - sdf_center)
    return normalize(n)


@luisa.func
def next_hit(pos, d):
    closest = inf
    normal = float3(0)
    c = float3(0)
    ray_march_dist = ray_march(pos, d)
    if ray_march_dist < dist_limit and ray_march_dist < closest:
        closest = ray_march_dist
        normal = sdf_normal(pos + d * closest)
        hit_pos = pos + d * closest
        t = int((hit_pos[0] + 10) * 1.1 + 0.5) % 3
        c = float3(
            0.4 + 0.3 * float(t == 0), 0.4 + 0.2 * float(t == 1), 0.4 + 0.3 * float(t == 2))
    return struct(closest=closest, normal=normal, c=c)


@luisa.func
def render(frame_index):
    set_block_size(16,8,1)
    res = float2(dispatch_size().xy)
    coord = dispatch_id().xy
    sampler = luisa.RandomSampler(int3(coord, frame_index))

    aspect_ratio = res.x / res.y
    pos = camera_pos
    uv = float2(coord.x + sampler.next(), res.y - 1 - coord.y + sampler.next())
    d = float3(2.0 * fov * uv / res.y - fov * float2(aspect_ratio, 1.0) - 1e-5, -1.0)
    d = normalize(d)

    throughput = float3(1)

    hit_light = 0.00

    for depth in range(max_ray_depth):
        hit = next_hit(pos, d)
        dist_to_light = intersect_light(pos, d)
        if dist_to_light < hit.closest:
            hit_light = 1.0
            break
        else:
            hit_pos = pos + hit.closest * d
            if length_squared(hit.normal) != 0:
                d = out_dir(hit.normal, sampler)
                pos = hit_pos + 1e-4 * d
                throughput *= hit.c
            else:
                break
    accum_color = float3(0)
    if frame_index != 0:
        accum_color = image.read(coord).xyz
    accum_color += throughput * hit_light
    image.write(coord, float4(accum_color, 1.0))


display = luisa.Texture2D.empty(*res, 4, float)

@luisa.func
def to_display(scale):
    coord = dispatch_id().xy
    accum_color = image.read(coord).xyz
    display.write(coord, float4(sqrt(accum_color * scale), 1.0))

ENABLE_DISPLAY = True
if ENABLE_DISPLAY:
    gui = luisa.GUI("SDF Path Tracer", res)
    frame_index = 0
    while gui.running():
        render(frame_index, dispatch_size=(*res, 1))
        frame_index += 1
        if frame_index % 4 == 0:
            to_display(0.24 / (1 + frame_index) / 0.084, dispatch_size=(*res, 1))
            gui.set_image(display)
            gui.show()

else:
    warm_up_spp = 128
    total_spp = 128
    interval = 4096
    frame_index = 0
    buffer = np.zeros([res[0] * res[1] * 4], dtype=np.float32)
    for i in range(warm_up_spp):
        render(i, dispatch_size=(*res, 1))
    display.copy_to(buffer)
    tic = perf_counter()
    for i in range(0, total_spp, interval):
        for j in range(interval):
            render(i + j + warm_up_spp, dispatch_size=(*res, 1))
        display.copy_to(buffer)
    toc = perf_counter()
    print("Speed = {:.2f} spp/s".format(total_spp / (toc - tic)))
