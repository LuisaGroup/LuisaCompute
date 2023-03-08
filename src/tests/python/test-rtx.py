from luisa import *
from luisa.builtin import *
from luisa.types import *
from sys import argv
import numpy as np
init()

res = 1024, 1024
image = Texture2D(*res, 4, float, storage="BYTE")

vertices = [
    float3(-0.5, -0.5, -2.0),
    float3(0.5, -0.5, -1.5),
    float3(0.0, 0.5, -1.0),
]
indices = np.array([0, 1, 2], dtype=np.int32)

vertex_buffer = Buffer(3, float3)
index_buffer = Buffer(3, int)
vertex_buffer.copy_from(vertices)
index_buffer.copy_from(indices)

accel = Accel()

accel.add(vertex_buffer, index_buffer)
accel.update()


@func
def halton(i, b):
    f = 1.0
    invB = 1.0 / b
    r = 0.0
    while i > 0:
        f *= invB
        r += f * (i % b)
        i = i // b
    return r


@func
def tea(v0, v1):
    s0 = 0
    for n in range(4):
        s0 += 0x9e3779b9
        v0 += ((v1 << 4) + 0xa341316c) ^ (v1 + s0) ^ ((v1 >> 5) + 0xc8013ea4)
        v1 += ((v0 << 4) + 0xad90777d) ^ (v0 + s0) ^ ((v0 >> 5) + 0x7e95761e)
    return v0


@func
def rand(f, p):
    i = tea(p.x, p.y) + f
    rx = halton(i, 2)
    ry = halton(i, 3)
    return float2(rx, ry)


@func
def raytracing_kernel(image, accel, frame_index):
    set_block_size(16, 16, 1)
    coord = dispatch_id().xy
    p = (float2(coord) + rand(frame_index, coord)) / \
        float2(dispatch_size().xy) * 2.0 - 1.0
    color = float3(0.3, 0.5, 0.7)
    ray = make_ray(
        float3(p * float2(1.0, -1.0), 0.0),  # origin
        float3(0.0, 0.0, -1.0),  # direction
        0.0,  # start distance
        1000.0)  # end distance
    q = accel.trace_all(ray, -1)
    hit = accel.trace_closest(ray, -1)
    # if hit.hit_type == 1:
    # if not hit.miss():
    if hit.hit_triangle():
        red = float3(1.0, 0.0, 0.0)
        green = float3(0.0, 1.0, 0.0)
        blue = float3(0.0, 0.0, 1.0)
        color = hit.interpolate(red, green, blue)
        # try ray distance
        # color = hit.ray_t * 0.5
    old = image.read(coord).xyz
    t = 1.0 / (frame_index + 1.0)
    image.write(coord, float4(lerp(old, color, t), 1.0))


frame_index = 0
gui = GUI("Test ray tracing", res)
while gui.running():
    raytracing_kernel(image, accel, frame_index, dispatch_size=(*res, 1))
    frame_index += 1
    gui.set_image(image)
    gui.show()
synchronize()
