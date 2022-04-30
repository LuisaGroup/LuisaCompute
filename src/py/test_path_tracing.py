import time

import numpy as np

import luisa
from luisa.mathtypes import *
from luisa.framerate import FrameRate
from luisa.window import Window
import dearpygui.dearpygui as dpg

from py.luisa.texture2d import Texture2DType

luisa.init("cuda")

Material = luisa.StructType(albedo=float3, emission=float3)

Onb = luisa.StructType(tangent=float3, binormal=float3, normal=float3)

@luisa.callable_method(Onb)
def to_world(self: luisa.ref(Onb), v: float3):
    return v.x * self.tangent + v.y * self.binormal + v.z * self.normal

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
def linear_to_srgb(x: float3):
    return clamp(select(1.055 * pow(x, 1.0 / 2.4) - 0.055,
                12.92 * x,
                x <= 0.00031308),
                0.0, 1.0)

@luisa.callable
def make_onb(normal: float3):
    binormal = normalize(ite(
            abs(normal.x) > abs(normal.z),
            make_float3(-normal.y, normal.x, 0.0),
            make_float3(0.0, -normal.z, normal.y)))
    tangent = normalize(cross(binormal, normal))
    result = Onb()
    Onb.tangent = tangent
    Onb.binormal = binormal
    Onb.normal = normal
    return result

@luisa.callable
def generate_ray():
    fov = radians(27.8) # TODO
    origin = make_float3(-0.01, 0.995, 5.0)
    pixel = origin + make_float3(p * tan(0.5 * fov), -1.0)
    direction = normalize(pixel - origin)
    return make_ray(origin, direction) # TODO

@luisa.callable
def cosine_sample_hemisphere(u: float2):
    r = sqrt(u.x)
    phi = 2.0 * 3.1415926 * u.y
    return make_float3(r * cos(phi), r * sin(phi), sqrt(1.0 - u.x))

@luisa.callable
def balanced_heuristic(pdf_a: float3, pdf_b: float3):
    return pdf_a / max(pdf_a + pdf_b, 1e-4)

@luisa.kernel
def raytracing_kernel(image: luisa.Texture2DType(float), accel: luisa.accel, resolution: int2, frame_index: int):
    set_block_size(16, 8, 1)
    coord = dispatch_id().xy
    frame_size = float(min(resolution.x, resolution.y))
    sampler = RandomSampler()
    sampler.__init__(make_int3(coord, frame_index))
    rx = sampler.next()
    ry = sampler.next()
    pixel = (make_float2(coord) + make_float2(rx, ry)) / frame_size * 2.0 - 1.0
    ray = generate_ray(pixel * make_float2(1.0, -1.0))
    radiance = make_float3(0.0)
    beta = make_float3(1.0)
    pdf_bsdf = 0.0
    light_position = make_float3(-0.24, 1.98, 0.16)
    light_u = make_float3(-0.24, 1.98, -0.22) - light_position
    light_v = make_float3(0.23, 1.98, 0.16) - light_position
    light_emission = make_float3(17.0, 12.0, 4.0)
    light_area = length(cross(light_u, light_v))
    light_normal = normalize(cross(light_u, light_v))

    for depth in range(5):
        # trace
        hit = accel.trace_closest(ray)
        if hit.miss():
            break
        triangle = heap.buffer<Triangle>(hit.inst).read(hit.prim)
        p0 = vertex_buffer.read(triangle.i0)
        p1 = vertex_buffer.read(triangle.i1)
        p2 = vertex_buffer.read(triangle.i2)
        p = hit.interpolate(p0, p1, p2)
        n = normalize(cross(p1 - p0, p2 - p0))
        cos_wi = dot(-ray.direction, n)
        if cos_wi < 1e-4:
            break
        material = material_buffer.read(hit.inst)

        # hit light
        if hit.inst == int(meshes.size() - 1):
            if depth == 0:
                radiance += light_emission
            else:
                pdf_light = length_squared(p - ray.origin) / (light_area * cos_wi)
                mis_weight = balanced_heuristic(pdf_bsdf, pdf_light)
                radiance += mis_weight * beta * light_emission
            break

        # sample light
        ux_light = sampler.next()
        uy_light = sampler.next()
        p_light = light_position + ux_light * light_u + uy_light * light_v
        pp = offset_ray_origin(p, n)
        pp_light = offset_ray_origin(p_light, light_normal)
        d_light = distance(pp, pp_light)
        wi_light = normalize(pp_light - pp)
        shadow_ray = make_ray(offset_ray_origin(pp, n), wi_light, 0.0, d_light)
        occluded = accel.trace_any(shadow_ray)
        cos_wi_light = dot(wi_light, n)
        cos_light = -dot(light_normal, wi_light)
        if (not occluded and cos_wi_light > 1e-4) and cos_light > 1e-4:
            pdf_light = (d_light * d_light) / (light_area * cos_light)
            pdf_bsdf = cos_wi_light * inv_pi
            mis_weight = balanced_heuristic(pdf_light, pdf_bsdf)
            bsdf = material.albedo * inv_pi * cos_wi_light
            radiance += beta * bsdf * mis_weight * light_emission / max(pdf_light, 1e-4)

        # sample BSDF
        onb = make_onb(n)
        ux = sampler.next()
        uy = sampler.next()
        new_direction = onb.to_world(cosine_sample_hemisphere(make_float2(ux, uy)))
        ray = make_ray(pp, new_direction)
        beta *= material.albedo
        pdf_bsdf = cos_wi * inv_pi

        # rr
        l = dot(make_float3(0.212671, 0.715160, 0.072169), beta)
        if l == 0.0:
            break
        q = max(l, 0.05)
        r = sampler.next()
        if r >= q:
            break
        beta *= 1.0 / q
    if any(isnan(radiance)):
        radiance = make_float3(0.0)
    image.write(dispatch_id().xy, make_float4(clamp(radiance, 0.0, 30.0), 1.0))

@luisa.kernel
def accumulate_kernel(accum_image: luisa.Texture2DType(float), curr_image: luisa.Texture2DType(float)):
    p = dispatch_id().xy
    accum = accum_image.read(p)
    curr = curr_image.read(p).xyz
    t = 1.0 / (accum.w + 1.0)
    accum_image.write(p, make_float4(lerp(accum.xyz, curr, t), accum.w + 1.0))

@luisa.callable
def aces_tonemapping(x: float3):
    a = 2.51
    b = 0.03
    c = 2.43
    d = 0.59
    e = 0.14
    return clamp((x * (a * x + b)) / (x * (c * x + d) + e), 0.0, 1.0)

@luisa.kernel
def clear_kernel(image: luisa.Texture2DType(float)):
    image.write(dispatch_id().xy, make_float4(0.0))

@luisa.kernel
def hdr2ldr_kernel(hdr_image: luisa.Texture2DType(float), ldr_image:luisa.Texture2DType(float), scale: float):
    coord = dispatch_id().xy
    hdr = hdr_image.read(coord)
    ldr = linear_to_srgb(hdr.xyz * scale)
    ldr_image.write(coord, make_float4(ldr, 1.0))

luisa.lcapi.log_level_error()

res = 1024, 1024
image = luisa.Texture2D(*res, 4, float)
accum_image = luisa.Texture2D(*res, 4, float)
ldr_image = luisa.Texture2D(*res, 4, float)
arr = np.zeros([*res, 4], dtype=np.float32)

frame_rate = FrameRate(10)
w = Window("Shader Toy", res, resizable=False, frame_rate=True)
w.set_background(arr, res)
dpg.draw_image("background", (0, 0), res, parent="viewport_draw")

clear_kernel(accum_image, dispatch_size=[*res, 1])


t0 = time.time()
frame_index = 0
sample_per_pass = 256
def update():
    global frame_index, arr
    t = time.time() - t0
    for i in range(sample_per_pass):
        raytracing_kernel(image, accel, make_int2(*res), frame_index, dispatch_size=(*res, 1))
        accumulate_kernel(accum_image, image, dispatch_size=[*res, 1])
        frame_index += 1
    hdr2ldr_kernel(image, ldr_image, 1.0, dispatch_size=[*res, 1])
    hdr2ldr_kernel(accum_image, ldr_image, 1.0, dispatch_size=[*res, 1])
    ldr_image.copy_to(arr)
    frame_rate.record()
    w.update_frame_rate(frame_rate.report() * sample_per_pass)
    print(frame_rate.report())
#     # w.update_frame_rate(dpg.get_frame_rate())

w.run(update)