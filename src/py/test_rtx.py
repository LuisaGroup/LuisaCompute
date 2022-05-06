import time

import numpy as np

import luisa
from luisa.mathtypes import *
from luisa.framerate import FrameRate
from luisa.window import Window
import dearpygui.dearpygui as dpg

from luisa.accel import make_ray
from luisa.texture2d import Texture2D

if len(argv) > 1:
    luisa.init(argv[1])
else:
    luisa.init("cuda")

res = 1280, 720
image = luisa.Buffer(res[0] * res[1], float4)
arr = np.zeros([*res, 4], dtype=np.float32)

vertices = [
    float3(-0.5, -0.5, -1.0),
    float3(0.5, -0.5, -1.0),
    float3(0.0, 0.5, -1.0),
]
indices = np.array([0, 1, 2], dtype=int)

vertex_buffer = luisa.Buffer(3, float3)
index_buffer = luisa.Buffer(3, int)
vertex_buffer.copy_from(vertices)
index_buffer.copy_from(indices)

accel = luisa.Accel()
mesh = luisa.Mesh(vertex_buffer, index_buffer)
accel.add(mesh)
accel.build()

@luisa.func
def linear_to_srgb(x: float3):
    return clamp(select(1.055 * x ** (1.0 / 2.4) - 0.055,
                12.92 * x,
                x <= 0.00031308),
                0.0, 1.0)

@luisa.func
def halton(i, b):
    f = 1.0
    invB = 1.0 / b
    r = 0.0
    while i > 0:
        f *= invB
        r += f * (i % b)
        i = i // b
    return r

@luisa.func
def tea(v0, v1):
    s0 = 0
    for n in range(4):
        s0 += 0x9e3779b9
        v0 += ((v1 << 4) + 0xa341316c) ^ (v1 + s0) ^ ((v1 >> 5) + 0xc8013ea4)
        v1 += ((v0 << 4) + 0xad90777d) ^ (v0 + s0) ^ ((v0 >> 5) + 0x7e95761e)
    return v0

@luisa.func
def rand(f, p):
    i = tea(p.x, p.y) + f
    rx = halton(i, 2)
    ry = halton(i, 3)
    return float2(rx, ry)

@luisa.func
def raytracing_kernel(image, accel, frame_index):
    coord = dispatch_id().xy
    p = (make_float2(coord) + rand(frame_index, coord)) / make_float2(dispatch_size().xy) * 2.0 - 1.0
    color = float3(0.3, 0.5, 0.7)
    ray = make_ray(
        make_float3(p * make_float2(1.0, -1.0), 1.0),
        make_float3(0.0, 0.0, -1.0), 0.0, 1e30)
    hit = accel.trace_closest(ray)
    if not hit.miss():
        red = float3(1.0, 0.0, 0.0)
        green = float3(0.0, 1.0, 0.0)
        blue = float3(0.0, 0.0, 1.0)
        color = hit.interpolate(red, green, blue)
    old = image.read(coord.y * dispatch_size().x + coord.x).xyz
    t = 1.0 / (frame_index + 1.0)
    image.write(coord.y * dispatch_size().x + coord.x, make_float4(lerp(old, color, t), 1.0))

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
        raytracing_kernel(image, accel, frame_index, dispatch_size=(*res, 1))
        frame_index += 1
    image.copy_to(arr)
    frame_rate.record()
    w.update_frame_rate(frame_rate.report())
    print(frame_rate.report())
#     # w.update_frame_rate(dpg.get_frame_rate())

w.run(update)