import time

import numpy as np

import luisa
from luisa.mathtypes import *
from luisa.framerate import FrameRate
from luisa.window import Window
import dearpygui.dearpygui as dpg

luisa.init("cuda")
res = 1280, 720
image = luisa.Texture2D(*res, 4, float)
display = luisa.Texture2D(*res, 4, float)
arr = np.zeros([*res, 4], dtype=np.float32)
max_ray_depth = 6
eps = 1e-4
inf = 1e10

fov = 0.23
dist_limit = 100

camera_pos = make_float3(0.0, 0.32, 3.7)
light_pos = make_float3(-1.5, 0.6, 0.3)
light_normal = make_float3(1.0, 0.0, 0.0)
light_radius = 2.0

next_hit_struct = luisa.StructType(closest=float, normal=float3, c=float3)

RandomSampler = luisa.StructType(state=int)

@luisa.callable_method(RandomSampler)
def __init__(self: luisa.ref(RandomSampler), p: int3):
    PRIME32_2 = 2246822519
    PRIME32_3 = 3266489917
    PRIME32_4 = 668265263
    PRIME32_5 = 374761393
    h32 =  p.z + PRIME32_5 + p.x * PRIME32_3
    h32 = PRIME32_4 * ((h32 << 17) | 0x0001ffff & (h32 >> (32 - 17)))
    h32 += p.y * PRIME32_3
    h32 = PRIME32_4 * ((h32 << 17) | 0x0001ffff & (h32 >> (32 - 17)))
    h32 = PRIME32_2 * (h32 ^ ((h32 >> 15) & 0x0001ffff))
    h32 = PRIME32_3 * (h32 ^ ((h32 >> 13) & 0x0007ffff))
    self.state = h32 ^ ((h32 >> 16) & 0x0000ffff)

@luisa.callable_method(RandomSampler)
def next(self: luisa.ref(RandomSampler)):
    lcg_a = 1664525
    lcg_c = 1013904223
    self.state = lcg_a * self.state + lcg_c
    return float(self.state & 0x00ffffff) * (1.0 / 0x01000000)


@luisa.callable
def intersect_light(pos: float3, d: float3):
    vdot = dot(-d, light_normal)
    dist = dot(d, light_pos - pos)
    dist_to_light = inf
    if vdot > 0 and dist > 0:
        D = dist / vdot
        dist_to_center = dot(light_pos - pos - D * d, light_pos - pos - D * d)
        if dist_to_center < light_radius * light_radius:
            dist_to_light = D
    return dist_to_light

@luisa.callable
def out_dir(n: float3, sampler: luisa.ref(RandomSampler)):
    u = make_float3(1.0, 0.0, 0.0)
    if abs(n.y) < 1 - eps:
        u = normalize(cross(n, make_float3(0.0, 1.0, 0.0)))
    v = cross(n, u)
    phi = 2 * 3.1415926 * sampler.next()
    ay = sqrt(sampler.next())
    ax = sqrt(1 - ay * ay)
    return ax * (cos(phi) * u + sin(phi) * v) + ay * n


@luisa.callable
def make_nested(f: float):
    f = f * 40.0
    # f = (1 - fract(f) if int(f) % 2 == 0 else fract(f)) if f < 0.0 else f
    # return (f - 0.2) * (1.0 / 40.0)

    i = int(f)
    if f < 0:
        if i % 2 != 0:
            f -= floor(f)
        else:
            f = floor(f) + 1 - f
    f = (f - 0.2) / 40
    return f

# https://www.iquilezles.org/www/articles/distfunctions/distfunctions.htm
@luisa.callable
def sdf(o: float3):
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


@luisa.callable
def ray_march(p: float3, d: float3):
    dist = 0.0
    for j in range(100):
        s = sdf(p + dist * d)
        if s <= 1e-6 or dist >= inf:
            break
        dist += s
    return min(inf, dist)


@luisa.callable
def sdf_normal(p: float3):
    d = 1e-3
    n = float3(0)
    sdf_center = sdf(p)
    for i in range(3):
        inc = p
        inc[i] += d
        n[i] = (1 / d) * (sdf(inc) - sdf_center)
    return normalize(n)


@luisa.callable
def next_hit(pos: float3, d: float3):
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
    result = next_hit_struct()
    result.closest = closest
    result.normal = normal
    result.c = c
    return result


@luisa.kernel
def render(image: luisa.Texture2DType(float), frame_index: int):
    set_block_size(16,8,1)
    res = make_float2(dispatch_size().xy)
    coord = dispatch_id().xy

    if frame_index == 0:
        image.write(dispatch_id().xy, make_float4(make_float3(0.0), 1.0))

    sampler = RandomSampler(make_int3(coord, frame_index))

    aspect_ratio = res.x / res.y
    pos = camera_pos
    ux = sampler.next()
    uy = sampler.next()
    uv = make_float2(dispatch_id().x + ux, dispatch_size().y - 1 - dispatch_id().y + uy)
    d = make_float3(
        2.0 * fov * uv / res.y - fov * make_float2(aspect_ratio, 1.0) - 1e-5, -1.0)
    d = normalize(d)

    throughput = make_float3(1.0, 1.0, 1.0)

    hit_light = 0.00

    for depth in range(max_ray_depth):
        result = next_hit(pos, d)
        closest = result.closest
        normal = result.normal
        c = result.c
        dist_to_light = intersect_light(pos, d)
        if dist_to_light < closest:
            hit_light = 1.0
            break
        else:
            hit_pos = pos + closest * d
            if dot(normal, normal) != 0:
                d = out_dir(normal, sampler)
                pos = hit_pos + 1e-4 * d
                throughput *= c
            else:
                break
    accum_color = image.read(dispatch_id().xy).xyz
    accum_color += throughput * hit_light
    image.write(dispatch_id().xy, make_float4(accum_color, 1.0))
    display.write(dispatch_id().xy, make_float4(sqrt(accum_color / (1.0 + frame_index) / 0.084 * 0.24), 1.0))

luisa.lcapi.log_level_error()

frame_rate = FrameRate(10)
w = Window("Shader Toy", res, resizable=False, frame_rate=True)
w.set_background(arr, res)
dpg.draw_image("background", (0, 0), res, parent="viewport_draw")

t0 = time.time()
frame_index = 0
def update():
    global frame_index, arr
    t = time.time() - t0
    for i in range(256):
        render(image, frame_index, dispatch_size=(*res, 1))
        frame_index += 1
    display.copy_to(arr)
    frame_rate.record()
    w.update_frame_rate(frame_rate.report())
    print(frame_rate.report())
#     # w.update_frame_rate(dpg.get_frame_rate())

w.run(update)