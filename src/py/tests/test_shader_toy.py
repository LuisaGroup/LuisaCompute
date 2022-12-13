from copyreg import dispatch_table
import math
import luisa
from luisa.mathtypes import *
import numpy as np
import time
import dearpygui.dearpygui as dpg
import array
from luisa.framerate import FrameRate
from luisa.window import Window
from sys import argv


if len(argv) > 1:
    luisa.init(argv[1])
else:
    luisa.init()

@luisa.func
def palette(d: float):
    return lerp(make_float3(0.2, 0.7, 0.9), make_float3(1.0, 0.0, 1.0), d)

@luisa.func
def rotate(p: float2, a: float):
    c = cos(a)
    s = sin(a)
    return make_float2(dot(p, make_float2(c, s)), dot(p, make_float2(-s, c)))

@luisa.func
def map(p: float3, time: float):
    q = p
    for i in range(8):
        t = time * 0.2
        q = make_float3(rotate(q.xz, t), q.y).xzy
        q = make_float3(rotate(q.xy, t * 1.89), q.z)
        q = make_float3(abs(q.x) - 0.5, q.y, abs(q.z) - 0.5)
    return dot(copysign(1.0, q), q) * 0.2

@luisa.func
def rm(ro: float3, rd: float3, time: float):
    t = 0.0
    col = make_float3(0.0)
    d = 0.0
    for i in range(64):
        p = ro + rd * t
        d = map(p, time) * 0.5
        if d < 0.02 or d > 100:
            break
        col += palette(length(p) * 0.1) / (400 * d)
        t += d
    return float4(col, 1.0 / (d * 100))

@luisa.func
def clear_kernel(image):
    coord = dispatch_id().xy
    rg = make_float2(coord) / make_float2(dispatch_size().xy)
    coordd = coord.x * dispatch_size().y + coord.y
    image.write(coordd * 4, 0.3)
    image.write(coordd * 4 + 1, 0.4)
    image.write(coordd * 4 + 2, 0.5)
    image.write(coordd * 4 + 3, 1.0)

@luisa.func
def main_kernel(image, time: float):
    xy = dispatch_id().xy
    coord = xy.y * dispatch_size().x + xy.x
    resolution = make_float2(dispatch_size().xy)
    uv = (make_float2(xy) - resolution * 0.5) / resolution.x
    ro = make_float3(rotate(make_float2(0, -50), time), 0.0).xzy
    cf = normalize(-ro)
    cs = normalize(cross(cf, make_float3(0.0, 1.0, 0.0)))
    cu = normalize(cross(cf, cs))
    uuv = ro + cf * 3.0 + uv.x * cs + uv.y * cu
    rd = normalize(uuv - ro)
    col = rm(ro, rd, time)
    image.write(coord * 4, col.x)
    image.write(coord * 4 + 1, col.y)
    image.write(coord * 4 + 2, col.z)
    image.write(coord * 4 + 3, 1.0)

@luisa.func
def naive_kernel(image, time: float):
    xy = dispatch_id().xy
    coord = xy.y * dispatch_size().x + xy.x
    scale = 1.0 / 1048576
    image.write(coord * 4, sin(time + scale * coord) * 0.5 + 0.5)
    image.write(coord * 4 + 1, sin(1.23432453245 * (time + scale * coord)) * 0.5 + 0.5)
    image.write(coord * 4 + 2, sin(2.32143241431 * (time + scale * coord)) * 0.5 + 0.5)
    image.write(coord * 4 + 3, 1.0)

image = luisa.Buffer(1024 * 1024 * 4, float)

arr = np.zeros([1024 * 1024 * 4], dtype=np.float32)

clear_kernel(image, dispatch_size=[1024, 1024, 1])

frame_rate = FrameRate(1)
w = Window("Shader Toy", (1024, 1024), resizable=False, frame_rate=True)
w.set_background(arr, (1024, 1024))
dpg.draw_image("background", (0, 0), (1024, 1024), parent="viewport_draw")

t0 = time.time()
def update():
    # frame_rate.record()
    t = time.time() - t0
    for i in range(1):
        main_kernel(image, t, dispatch_size=(1024, 1024, 1))
    image.copy_to(arr)
    # w.update_frame_rate(frame_rate.report())
    w.update_frame_rate(dpg.get_frame_rate() * 16)


def null():
    # frame_rate.record()
    w.update_frame_rate(dpg.get_frame_rate())

w.run(update)
