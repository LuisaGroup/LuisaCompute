import time

import numpy as np
from sys import argv

import luisa
from luisa.mathtypes import *
from luisa.framerate import FrameRate
from luisa.window import Window
import dearpygui.dearpygui as dpg

import cornell_box
from luisa.accel import make_ray, offset_ray_origin


if len(argv) > 1:
    luisa.init(argv[1])
else:
    luisa.init("cuda")


Material = luisa.StructType(albedo=float3, emission=float3)

Onb = luisa.StructType(tangent=float3, binormal=float3, normal=float3)

heap = luisa.BindlessArray()
vertex_buffer = luisa.Buffer(len(cornell_box.vertices), float3)
vertex_arr = [[*item, 0.0] for item in cornell_box.vertices]
vertex_arr = np.array(vertex_arr, dtype=np.float32)
vertex_buffer.copy_from(vertex_arr)
material_arr = [
    [0.725, 0.71, 0.68, 0.0], [0.0, 0.0, 0.0, 0.0],
    [0.725, 0.71, 0.68, 0.0], [0.0, 0.0, 0.0, 0.0],
    [0.725, 0.71, 0.68, 0.0], [0.0, 0.0, 0.0, 0.0],
    [0.14, 0.45, 0.091, 0.0], [0.0, 0.0, 0.0, 0.0],
    [0.63, 0.065, 0.05, 0.0], [0.0, 0.0, 0.0, 0.0],
    [0.725, 0.71, 0.68, 0.0], [0.0, 0.0, 0.0, 0.0],
    [0.725, 0.71, 0.68, 0.0], [0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0], [17.0, 12.0, 4.0, 0.0],
]
material_buffer = luisa.Buffer(len(material_arr), float3)
material_arr = np.array(material_arr, dtype=np.float32)
material_buffer.copy_from(material_arr)

mesh_cnt = 0
accel = luisa.Accel()
for mesh in cornell_box.faces:
    indices = []
    for item in mesh:
        assert(len(item) == 4)
        for x in [0, 1, 2, 0, 2, 3]:
            indices.append(item[x])
    triangle_buffer = luisa.Buffer(len(indices), int)
    triangle_buffer.copy_from(np.array(indices, dtype=np.int32))
    heap.emplace(mesh_cnt, triangle_buffer)
    mesh = luisa.Mesh(vertex_buffer, triangle_buffer)
    accel.add(mesh)
    mesh_cnt += 1
accel.update()
heap.update()

@luisa.func
def to_world(self, v: float3):
    return v.x * self.tangent + v.y * self.binormal + v.z * self.normal
Onb.add_method(to_world, "to_world")

from luisa.util import RandomSampler

@luisa.func
def linear_to_srgb(x: float3):
    return clamp(select(1.055 * x ** (1.0 / 2.4) - 0.055,
                12.92 * x,
                x <= 0.00031308),
                0.0, 1.0)

@luisa.func
def make_onb(normal: float3):
    binormal = normalize(select(
            make_float3(0.0, -normal.z, normal.y),
            make_float3(-normal.y, normal.x, 0.0),
            abs(normal.x) > abs(normal.z)))
    tangent = normalize(cross(binormal, normal))
    result = Onb()
    result.tangent = tangent
    result.binormal = binormal
    result.normal = normal
    return result

@luisa.func
def generate_ray(p):
    fov = 27.8 / 180 * 3.1415926
    origin = make_float3(-0.01, 0.995, 5.0)
    pixel = origin + make_float3(p * tan(0.5 * fov), -1.0)
    direction = normalize(pixel - origin)
    return make_ray(origin, direction, 0.0, 1e30) # TODO

@luisa.func
def cosine_sample_hemisphere(u: float2):
    r = sqrt(u.x)
    phi = 2.0 * 3.1415926 * u.y
    return make_float3(r * cos(phi), r * sin(phi), sqrt(1.0 - u.x))

@luisa.func
def balanced_heuristic(pdf_a, pdf_b):
    return pdf_a / max(pdf_a + pdf_b, 1e-4)

@luisa.func
def raytracing_kernel(image, accel, resolution, frame_index):
    set_block_size(8, 8, 1)
    coord = dispatch_id().xy
    frame_size = float(min(resolution.x, resolution.y))
    sampler = RandomSampler(make_int3(coord, frame_index))
    rx = sampler.next()
    ry = sampler.next()
    pixel = (make_float2(coord) + make_float2(rx, ry)) / frame_size * 2.0 - 1.0
    ray = generate_ray(pixel * make_float2(1.0, -1.0))
    radiance = make_float3(0.0)
    beta = make_float3(1.0)
    pdf_bsdf = 1e30
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
        i0 = heap.buffer_read(int, hit.inst, hit.prim * 3 + 0)
        i1 = heap.buffer_read(int, hit.inst, hit.prim * 3 + 1)
        i2 = heap.buffer_read(int, hit.inst, hit.prim * 3 + 2)
        p0 = vertex_buffer.read(i0)
        p1 = vertex_buffer.read(i1)
        p2 = vertex_buffer.read(i2)
        p = hit.interpolate(p0, p1, p2)
        n = normalize(cross(p1 - p0, p2 - p0))
        cos_wi = dot(-ray.get_dir(), n)
        if cos_wi < 1e-4:
            break
        material = Material()
        material.albedo = material_buffer.read(hit.inst * 2 + 0)
        material.emission = material_buffer.read(hit.inst * 2 + 1)

        # hit light
        if hit.inst == int(mesh_cnt - 1):
            if depth == 0:
                radiance += light_emission
            else:
                pdf_light = length_squared(p - ray.get_origin()) / (light_area * cos_wi)
                mis_weight = balanced_heuristic(pdf_bsdf, pdf_light)
                radiance += mis_weight * beta * light_emission
            break

        # sample light
        ux_light = sampler.next()
        uy_light = sampler.next()
        p_light = light_position + ux_light * light_u + uy_light * light_v
        pp = offset_ray_origin(p, n)
        pp_light = offset_ray_origin(p_light, light_normal)
        d_light = length(pp - pp_light)
        wi_light = normalize(pp_light - pp)
        shadow_ray = make_ray(offset_ray_origin(pp, n), wi_light, 0.0, d_light)
        occluded = accel.trace_any(shadow_ray)
        cos_wi_light = dot(wi_light, n)
        cos_light = -dot(light_normal, wi_light)
        if ((not occluded and cos_wi_light > 1e-4) and cos_light > 1e-4):
            pdf_light = (d_light * d_light) / (light_area * cos_light)
            pdf_bsdf = cos_wi_light * (1 / 3.1415926)
            mis_weight = balanced_heuristic(pdf_light, pdf_bsdf)
            bsdf = material.albedo * (1 / 3.1415926) * cos_wi_light
            # radiance += beta * bsdf * light_emission
            radiance += beta * bsdf * mis_weight * light_emission / max(pdf_light, 1e-4)

        # sample BSDF
        onb = make_onb(n)
        ux = sampler.next()
        uy = sampler.next()
        new_direction = onb.to_world(cosine_sample_hemisphere(make_float2(ux, uy)))
        ray = make_ray(pp, new_direction, 0.0, 1e30)
        # bsdf = material.albedo / 3.1415926 * cos_wi
        # beta *= bsdf / pdf_bsdf
        beta *= material.albedo
        pdf_bsdf = cos_wi * (1 / 3.1415926)

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

@luisa.func
def accumulate_kernel(accum_image, curr_image):
    p = dispatch_id().xy
    accum = accum_image.read(p)
    curr = curr_image.read(p).xyz
    t = 1.0 / (accum.w + 1.0)
    accum_image.write(p, make_float4(lerp(accum.xyz, curr, t), accum.w + 1.0))

@luisa.func
def aces_tonemapping(x: float3):
    a = 2.51
    b = 0.03
    c = 2.43
    d = 0.59
    e = 0.14
    return clamp((x * (a * x + b)) / (x * (c * x + d) + e), 0.0, 1.0)

@luisa.func
def clear_kernel(image):
    image.write(dispatch_id().xy, make_float4(0.0))

@luisa.func
def hdr2ldr_kernel(hdr_image, ldr_image, scale: float):
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
    hdr2ldr_kernel(accum_image, ldr_image, 1.0, dispatch_size=[*res, 1])
    ldr_image.copy_to(arr)
    frame_rate.record(sample_per_pass)
    w.update_frame_rate(frame_rate.report())
    print(frame_rate.report())
#     # w.update_frame_rate(dpg.get_frame_rate())

w.run(update)
