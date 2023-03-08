from luisa import *
from luisa.types import *
from luisa.builtin import *
import numpy as np
init()


@func
def _palette(d: float):
    return lerp(float3(0.2, 0.7, 0.9), float3(1., 0., 1.), d)


@func
def _rotate(p: float2, a: float):
    c = cos(a)
    s = sin(a)
    return float2(dot(p, float2(c, s)), dot(p, float2(-s, c)))


@func
def _map(p: float3, time: float):
    for i in range(8):
        t = time * 0.2
        p = float3(_rotate(p.xz, t), p.y).xzy
        p = float3(_rotate(p.xy, t * 1.89), p.z)
        p = float3(abs(p.x) -0.5, p.y, abs(p.z) - 0.5)
    return dot(copysign(1., p), p) * .2


@func
def _rm(ro: float3, rd: float3, time: float):
    t = 0.
    col = float3()
    d = 0.
    for i in range(64):
        p = ro + rd * t
        d = _map(p, time) * .5
        if d < 0.02 or d > 100:
            break
        col += _palette(length(p) * 0.1) / (400. * d)
        t += d
    return float4(col, 1. / (d * 100.))


@func
def clear_kernel(image):
    coord = dispatch_id().xy
    rg = float2(coord) / float2(dispatch_size().xy)
    image.write(coord, float4(0.3, 0.4, 0.5, 1.))


@func
def main_kernel(image, time):
    xy = dispatch_id().xy
    resolution = float2(dispatch_size().xy)
    uv = (float2(xy) - resolution * 0.5) / resolution.x
    ro = float3(_rotate(float2(0.0, -50.), time), 0.0).xzy
    cf = normalize(-ro)
    cs = normalize(cross(cf, float3(0, 1, 0)))
    cu = normalize(cross(cf, cs))
    uuv = ro + cf * 3.0 + uv.x * cs + uv.y * cu
    rd = normalize(uuv - ro)
    col = _rm(ro, rd, time)
    color = col.xyz
    alpha = col.w
    old = image.read(xy).xyz
    accum = lerp(color, old, alpha)
    image.write(xy, float4(accum, 1.0))


res = 1024, 1024
image = Texture2D(*res, 4, float, storage="BYTE")

gui = GUI("Test ray tracing", res)
clear_kernel(image, dispatch_size=(*res, 1))
time = 0.0
while gui.running():
    main_kernel(image, time, dispatch_size=(*res, 1))
    gui.set_image(image)
    # use seconds
    time += gui.show() / 1000.0
synchronize()
