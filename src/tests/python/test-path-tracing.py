from luisa import *
from luisa.builtin import *
from luisa.types import *
from luisa.util import *
import aces

import time
import cornell_box
import numpy as np
init()

Material = StructType(albedo=float3, emission=float3)
Onb = StructType(tangent=float3, binormal=float3, normal=float3)


@func
def to_world(self, v: float3):
    return v.x * self.tangent + v.y * self.binormal + v.z * self.normal


Onb.add_method(to_world, "to_world")

@func
def make_onb(normal: float3):
    binormal = normalize(select(
        float3(0.0, -normal.z, normal.y),
        float3(-normal.y, normal.x, 0.0),
        abs(normal.x) > abs(normal.z)))
    tangent = normalize(cross(binormal, normal))
    result = Onb()
    result.tangent = tangent
    result.binormal = binormal
    result.normal = normal
    return result


@func
def generate_ray(p):
    fov = 27.8 / 180 * 3.1415926
    origin = float3(-0.01, 0.995, 5.0)
    pixel = origin + float3(p * tan(0.5 * fov), -1.0)
    direction = normalize(pixel - origin)
    return make_ray(origin, direction, 0.0, 1e30)


@func
def cosine_sample_hemisphere(u: float2):
    r = sqrt(u.x)
    phi = 2.0 * 3.1415926 * u.y
    return float3(r * cos(phi), r * sin(phi), sqrt(1.0 - u.x))


@func
def balanced_heuristic(pdf_a, pdf_b):
    return pdf_a / max(pdf_a + pdf_b, 1e-4)


enable_aces = True


@func
def raytracing_kernel(image, seed_image, accel, heap, resolution, vertex_buffer, material_buffer, mesh_cnt, frame_index):
    set_block_size(8, 8, 1)
    coord = dispatch_id().xy
    frame_size = float(min(resolution.x, resolution.y))
    if frame_index == 0:
        sampler = make_random_sampler(coord.x, coord.y)
    else:
        sampler = RandomSampler(seed_image.read(coord))
    radiance = float3(0)
    light_position = float3(-0.24, 1.98, 0.16)
    light_u = float3(-0.24, 1.98, -0.22) - light_position
    light_v = float3(0.23, 1.98, 0.16) - light_position
    light_emission = float3(17.0, 12.0, 4.0)
    if enable_aces:
        light_emission = aces.srgb_to_acescg(light_emission)
    light_area = length(cross(light_u, light_v))
    light_normal = normalize(cross(light_u, light_v))
    rx = sampler.next()
    ry = sampler.next()
    pixel = (float2(coord) + float2(rx, ry)) / frame_size * 2.0 - 1.0
    ray = generate_ray(pixel * float2(1.0, -1.0))
    beta = float3(1.0)
    pdf_bsdf = 1e30
    for depth in range(5):
        # trace
        hit = accel.trace_closest(ray, -1)
        # advanced usage: rayquery
        # query = accel.trace_all(ray, -1)
        # while(query.proceed()):
        #     if query.is_triangle():
        #         query.commit_triangle()
        #     else:
        #         # no procedural in corner box
        #         continue
        # hit = query.committed_hit()
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
        if enable_aces:
            material.albedo = aces.srgb_to_acescg(material.albedo)
            material.emission = aces.srgb_to_acescg(material.emission)
        # hit light
        if hit.inst == int(mesh_cnt - 1):
            if depth == 0:
                radiance += light_emission
            else:
                pdf_light = length_squared(
                    p - ray.get_origin()) / (light_area * cos_wi)
                mis_weight = float(balanced_heuristic(pdf_bsdf, pdf_light))
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
        occluded = accel.trace_any(shadow_ray, -1)
        cos_wi_light = dot(wi_light, n)
        cos_light = -dot(light_normal, wi_light)
        if ((not occluded and cos_wi_light > 1e-4) and cos_light > 1e-4):
            pdf_light = (d_light * d_light) / (light_area * cos_light)
            pdf_bsdf = cos_wi_light * (1 / 3.1415926)
            mis_weight = float(balanced_heuristic(pdf_light, pdf_bsdf))
            bsdf = float3(material.albedo * (1 / 3.1415926) * cos_wi_light)
            # radiance += beta * bsdf * light_emission
            radiance += beta * bsdf * mis_weight * \
                light_emission / float(max(pdf_light, 1e-4))

        # sample BSDF
        onb = make_onb(n)
        ux = sampler.next()
        uy = sampler.next()
        new_direction = onb.to_world(
            cosine_sample_hemisphere(float2(ux, uy)))
        ray = make_ray(pp, new_direction, 0.0, 1e30)
        # bsdf = material.albedo / 3.1415926 * cos_wi
        # beta *= bsdf / pdf_bsdf
        beta *= material.albedo
        pdf_bsdf = cos_wi * (1 / 3.1415926)

        # rr
        l = dot(float3(0.212671, 0.715160, 0.072169), beta)
        if l == 0.0:
            break
        q = max(l, 0.05)
        r = sampler.next()
        if r >= q:
            break
        beta *= float(1.0 / q)
        if any(isnan(radiance)):
            radiance = float3(0.0)
    seed_image.write(coord, sampler.state)
    if enable_aces:
        radiance = aces.acescg_to_srgb(radiance)
    image.write(coord, float4(
        clamp(radiance, 0.0, 30.0), 1.0))


@func
def accumulate_kernel(accum_image, curr_image):
    p = dispatch_id().xy
    accum = accum_image.read(p)
    curr = curr_image.read(p).xyz
    t = 1.0 / (accum.w + 1.0)
    accum_image.write(p, float4(lerp(accum.xyz, curr, t), accum.w + 1.0))


@func
def aces_tonemapping(x: float3):
    a = 2.51
    b = 0.03
    c = 2.43
    d = 0.59
    e = 0.14
    return clamp((x * (a * x + b)) / (x * (c * x + d) + e), 0.0, 1.0)


@func
def clear_kernel(image):
    image.write(dispatch_id().xy, float4(0.0))


@func
def hdr2ldr_kernel(hdr_image, ldr_image, scale: float):
    coord = dispatch_id().xy
    hdr = hdr_image.read(coord)
    ldr = aces.linear_to_srgb(hdr.xyz * scale)
    ldr_image.write(coord, float4(ldr, 1.0))


heap = BindlessArray()
vertex_buffer = Buffer(len(cornell_box.vertices), float3)
vertex_arr = [[*item, 0.0] for item in cornell_box.vertices]
vertex_arr = np.array(vertex_arr, dtype=np.float32)
vertex_buffer.copy_from(vertex_arr)
material_arr = [
    [1, 1, 1, 0.0], [0.0, 0.0, 0.0, 0.0],
    [1, 1, 1, 0.0], [0.0, 0.0, 0.0, 0.0],
    [1, 1, 1, 0.0], [0.0, 0.0, 0.0, 0.0],
    [0, 1, 0, 0.0], [0.0, 0.0, 0.0, 0.0],
    [1, 0, 0, 0.0], [0.0, 0.0, 0.0, 0.0],
    [1, 0, 0, 0.0], [0.0, 0.0, 0.0, 0.0],
    [0, 1, 0, 0.0], [0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0], [10.0, 10.0, 10.0, 0.0],
]
material_buffer = Buffer(len(material_arr), float3)
material_arr = np.array(material_arr, dtype=np.float32)
material_buffer.copy_from(material_arr)

mesh_cnt = 0
accel = Accel()
for mesh in cornell_box.faces:
    indices = []
    for item in mesh:
        assert (len(item) == 4)
        for x in [0, 1, 2, 0, 2, 3]:
            indices.append(item[x])
    if mesh_cnt == 5:
        inst = 64
    else:
        inst = 128
    triangle_buffer = Buffer(len(indices), int)
    triangle_buffer.copy_from(np.array(indices, dtype=np.int32))
    heap.emplace(mesh_cnt, triangle_buffer)
    accel.add(vertex_buffer, triangle_buffer, visibility_mask=inst)
    mesh_cnt += 1
accel.update()
heap.update()
res = 1024, 1024
image = Image2D(*res, 4, float)
accum_image = Image2D(*res, 4, float)
seed_image = Image2D(*res, 1, uint, storage="INT")
ldr_image = Image2D(*res, 4, float, storage="BYTE")

clear_kernel(accum_image, dispatch_size=[*res, 1])


t0 = time.time()
frame_index = 0


def sample():
    global frame_index, image, accel, res
    raytracing_kernel(image, seed_image, accel, heap, make_int2(
        *res), vertex_buffer, material_buffer, mesh_cnt, frame_index, dispatch_size=(*res, 1))
    accumulate_kernel(accum_image, image, dispatch_size=[*res, 1])
    hdr2ldr_kernel(accum_image, ldr_image, 1.0, dispatch_size=[*res, 1])
    frame_index += 1


gui = GUI("Test cornell box", res)
while gui.running():
    sample()
    gui.set_image(ldr_image)
    gui.show()
synchronize()