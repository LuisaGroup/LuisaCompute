from sys import argv
import luisa
from luisa.mathtypes import *
from math import pi
from PIL import Image
from luisa.util import RandomSampler
from time import perf_counter

from disney import *

if len(argv) > 1:
    luisa.init(argv[1])
else:
    luisa.init()
luisa.log_level_verbose()
from water import models, resolution, max_depth, rr_depth, \
                   const_env_light, camera_pos, camera_dir, camera_up, camera_fov, __name__ as outfile
# max_depth, rr_depth = 8, 2
max_depth, rr_depth = 16, 5
# models: (filename, mat, [emission], [transform])
from parseobj import parseobj

if camera_fov > pi: # likely to be in degrees; convert to radian
    camera_fov *= pi / 180
camera_right = luisa.lcapi.normalize(luisa.lcapi.cross(camera_dir, camera_up))
camera_up = luisa.lcapi.normalize(luisa.lcapi.cross(camera_right, camera_dir))




# load scene

meshes = [] # each mesh
heapindex = {} # idx -> buffer
emission = [] # emission of each mesh
light_meshid = [] # list of emissive meshes
materials = []
tricount = [] # number of triangles in each mesh
has_texture_list = []

def flatten_list(a):
    s = []
    for x in a:
        s += x
    return s

VertInfo = luisa.ArrayType(dtype=float, size=8)
@luisa.func
def vert_v(info: VertInfo):
    return float3(info[0], info[1], info[2])
@luisa.func
def vert_vt(info: VertInfo):
    return float2(info[3], info[4])
@luisa.func
def vert_vn(info: VertInfo):
    return float3(info[5], info[6], info[7])


for idx, model in enumerate(models):
    filename, material = model[0:2]
    # texture?
    has_texture = 0
    if type(material) is tuple:
        if len(material) == 1:
            material = material[0]
        elif len(material) == 2:
            texture, material = material
            heapindex[idx+4096] = texture
            has_texture = 1
            material.base_color = float3(0.5)
    has_texture_list.append(has_texture)
    materials.append(material)
    # load mesh
    print("loading", filename)
    v,vt,vn,f = parseobj(open(filename))
    assert len(v) == len(vn) > 0
    if len(vt) == 0:
        vt = [[0,0]]*len(v)
    else:
        assert len(vt)==len(v)
    # add mesh
    vertex_buffer = luisa.buffer([VertInfo([*v[i], *vt[i], *vn[i]]) for i in range(len(v))])
    tricount.append(len(f))
    triangle_buffer = luisa.buffer(flatten_list(f))
    mesh = luisa.Mesh(vertex_buffer, triangle_buffer)
    meshes.append(mesh)
    heapindex[idx*2+1] = vertex_buffer
    heapindex[idx*2+0] = triangle_buffer
    # emission
    if len(model)>2 and model[2] is not None and model[2] != float3(0):
        assert type(model[2]) is float3
        emission.append(model[2])
        light_meshid.append(idx)
    else:
        emission.append(float3(0))
    # transform
    if len(model)>3:
        transform = model[3]
        assert type(transform) is float4x4
        meshes[-1] = (mesh, transform)

print("lights", light_meshid)
accel = luisa.accel(meshes)
heap = luisa.bindless_array(heapindex)
material_buffer = luisa.buffer(materials)
emission_buffer = luisa.buffer(emission)
tricount_buffer = luisa.buffer(tricount)
has_texture_buffer = luisa.buffer(has_texture_list)
light_array = luisa.array(light_meshid)
light_count = len(light_meshid)



@luisa.func
def sample_uniform_triangle(u: float2):
    uv = make_float2(0.5 * u.x, -0.5 * u.x + u.y) \
         if u.x < u.y else \
         make_float2(-0.5 * u.y + u.x, 0.5 * u.y)
    return make_float3(uv, 1.0 - uv.x - uv.y)

@luisa.func
def sample_uniform_sphere(u: float2):
    z = 1.0 - 2.0 * u.x;
    r = sqrt(max(1.0 - z * z, 0.0))
    phi = 2.0 * pi * u.y
    return make_float3(r * cos(phi), r * sin(phi), z)

env_prob = 0.3

@luisa.func
def mesh_light_sampled_pdf(p, origin, inst, p0, p1, p2):
    n = light_count
    n1 = tricount_buffer.read(inst)
    wi_light = normalize(p - origin)
    c = cross(p1 - p0, p2 - p0)
    light_normal = normalize(c)
    cos_light = -dot(light_normal, wi_light)
    sqr_dist = length_squared(p - origin)
    area = length(c) / 2
    pdf = (1.0 - env_prob) * sqr_dist / (n * n1 * area * abs(cos_light))
    return pdf

@luisa.func
def env_light_sampled_pdf(wi):
    return env_prob / (4 * pi)

# returns: (wi, dist, pdf, eval)
@luisa.func
def sample_light(origin, sampler):
    u = sampler.next()
    if u < env_prob:
        emission = const_env_light
        wi = sample_uniform_sphere(sampler.next2f())
        pdf = env_prob / (4 * pi)
        return struct(wi=wi, dist=1e30, pdf=pdf, eval=emission)
    else:
        u_remapped = (u - env_prob) / (1.0 - env_prob)
        n = light_count
        inst = light_array[clamp(int(u_remapped * n), 0, n-1)]
        n1 = tricount_buffer.read(inst)
        prim = clamp(int(sampler.next() * n1), 0, n1-1)
        i0 = heap.buffer_read(int, inst * 2, prim * 3 + 0)
        i1 = heap.buffer_read(int, inst * 2, prim * 3 + 1)
        i2 = heap.buffer_read(int, inst * 2, prim * 3 + 2)
        p0 = vert_v(heap.buffer_read(VertInfo, inst * 2 + 1, i0))
        p1 = vert_v(heap.buffer_read(VertInfo, inst * 2 + 1, i1))
        p2 = vert_v(heap.buffer_read(VertInfo, inst * 2 + 1, i2))
        # apply transform
        transform = accel.instance_transform(inst)
        p0 = (transform * float4(p0, 1.0)).xyz
        p1 = (transform * float4(p1, 1.0)).xyz
        p2 = (transform * float4(p2, 1.0)).xyz
        abc = sample_uniform_triangle(sampler.next2f())
        p = abc.x * p0 + abc.y * p1 + abc.z * p2 # point on light
        emission = emission_buffer.read(inst)
        # calculating pdf (avoid calling mesh_light_sampled_pdf to save some redundant computation)
        wi_light = normalize(p - origin)
        c = cross(p1 - p0, p2 - p0)
        light_normal = normalize(c)
        cos_light = -dot(light_normal, wi_light)
        emission = emission if cos_light > 1e-4 else float3(0)
        sqr_dist = length_squared(p - origin)
        area = length(c) / 2
        pdf = (1.0 - env_prob) * sqr_dist / (n * n1 * area * cos_light)
        return struct(wi=wi_light, dist=0.9999*sqrt(sqr_dist), pdf=pdf, eval=emission)


Onb = luisa.StructType(tangent=float3, binormal=float3, normal=float3)
@luisa.func
def to_world(self, v: float3):
    return v.x * self.tangent + v.y * self.binormal + v.z * self.normal
Onb.add_method(to_world, "to_world")

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
def generate_camera_ray(sampler, resolution):
    coord = dispatch_id().xy
    frame_size = float(min(resolution.x, resolution.y))
    pixel = ((make_float2(coord) + sampler.next2f()) * 2.0 - float2(resolution)) / frame_size # remapped to [-1,1] in shorter axis
    d = make_float3(pixel * make_float2(1.0, -1.0) * tan(0.5 * camera_fov), 1.0)
    direction = normalize(camera_right * d.x + camera_up * d.y + camera_dir * d.z)
    return make_ray(camera_pos, direction, 0.0, 1e30) # TODO


@luisa.func
def cosine_sample_hemisphere(u: float2):
    r = sqrt(u.x)
    phi = 2.0 * 3.1415926 * u.y
    return make_float3(r * cos(phi), r * sin(phi), sqrt(1.0 - u.x))


@luisa.func
def balanced_heuristic(pdf_a, pdf_b):
    return pdf_a / max(pdf_a + pdf_b, 1e-4)

@luisa.func
def srgb_to_linear(x: float3):
    return select(pow((x + 0.055) * (1.0 / 1.055), 2.4),
               x * (1. / 12.92),
                x <= 0.04045)


@luisa.func
def path_tracer(accum_image, accel, resolution, frame_index):
    set_block_size(8, 8, 1)
    coord = dispatch_id().xy
    sampler = RandomSampler(make_int3(coord, frame_index))
    ray = generate_camera_ray(sampler, resolution)

    radiance = make_float3(0.0)
    beta = make_float3(1.0)
    pdf_bsdf = 1e30

    for depth in range(max_depth):
        # trace
        hit = accel.trace_closest(ray)
        # miss: evaluate environment light
        if hit.miss():
            emission = const_env_light
            if depth == 0:
                radiance += emission
            else:
                pdf_env = env_light_sampled_pdf(ray.get_dir())
                mis_weight = balanced_heuristic(pdf_bsdf, pdf_env)
                radiance += mis_weight * beta * emission
            break
        # fetch hit triangle info
        i0 = heap.buffer_read(int, hit.inst * 2, hit.prim * 3 + 0)
        i1 = heap.buffer_read(int, hit.inst * 2, hit.prim * 3 + 1)
        i2 = heap.buffer_read(int, hit.inst * 2, hit.prim * 3 + 2)
        vert_info0 = heap.buffer_read(VertInfo, hit.inst * 2 + 1, i0)
        vert_info1 = heap.buffer_read(VertInfo, hit.inst * 2 + 1, i1)
        vert_info2 = heap.buffer_read(VertInfo, hit.inst * 2 + 1, i2)
        p0 = vert_v(vert_info0)
        p1 = vert_v(vert_info1)
        p2 = vert_v(vert_info2)
        vn0 = vert_vn(vert_info0)
        vn1 = vert_vn(vert_info1)
        vn2 = vert_vn(vert_info2)
        vn = hit.interpolate(vn0, vn1, vn2)
        # apply transform
        transform = accel.instance_transform(hit.inst)
        p0 = (transform * float4(p0, 1.0)).xyz
        p1 = (transform * float4(p1, 1.0)).xyz
        p2 = (transform * float4(p2, 1.0)).xyz
        # get hit position, onb and albedo (surface color)
        p = hit.interpolate(p0, p1, p2)
        ng = normalize(cross(p1 - p0, p2 - p0))
        n = normalize(inverse(transpose(make_float3x3(transform))) * vn)
        wo = -ray.get_dir()
        material = material_buffer.read(hit.inst)
        has_texture = has_texture_buffer.read(hit.inst)
        if has_texture != 0:
            vt0 = vert_vt(vert_info0)
            vt1 = vert_vt(vert_info1)
            vt2 = vert_vt(vert_info2)
            uv = hit.interpolate(vt0, vt1, vt2)
            uv.y = 1.0 - uv.y
            srgb = heap.texture2d_sample(hit.inst + 4096, uv).xyz
            material.base_color = srgb_to_linear(srgb)
        emission = emission_buffer.read(hit.inst)
        if material.specular_transmission == 0.0:
            if dot(wo, n) < 0:
                n = -n
        onb = make_onb(n)

        # hit light
        if any(emission != float3(0)):
            if depth == 0:
                radiance += emission
            else:
                pdf_light = mesh_light_sampled_pdf(p, ray.get_origin(), hit.inst, p0, p1, p2)
                mis_weight = balanced_heuristic(pdf_bsdf, pdf_light)
                # mis_weight = 0.0
                radiance += mis_weight * beta * emission
            break

        # sample light
        light = sample_light(p, sampler) # (wi, dist, pdf, eval)
        shadow_ray = make_ray(p, light.wi, 1e-4, light.dist)
        occluded = accel.trace_any(shadow_ray)
        cos_wi_light = dot(light.wi, n)

        # if dispatch_id().y == 200:
        #     accum = accum_image.read(coord).xyz
        #     accum_image.write(coord, make_float4(accum + float3(0.5, 0.0, 0.0), 1.0))
        #     print(light, occluded, cos_wi_light)
        #     return

        # DEBUG override glass # material.specular_transmission == 0.0 and 
        if not occluded:
            bsdf = disney_brdf(material, onb.normal, wo, light.wi, onb.binormal, onb.tangent)
            pdf_bsdf = disney_pdf(material, onb.normal, wo, light.wi, onb.binormal, onb.tangent)
            mis_weight = balanced_heuristic(light.pdf, pdf_bsdf)
            # mis_weight = 1.0
            radiance += beta * bsdf * cos_wi_light * mis_weight * light.eval / max(light.pdf, 1e-4)

        # sample BSDF (pdf, w_i, brdf)
        sample = sample_disney_brdf(material, onb.normal, wo, onb.binormal, onb.tangent, sampler)
        # t3 = cosine_sample_hemisphere(sampler.next2f())
        # w_i = t3.x * onb.binormal + t3.y * onb.tangent + t3.z * (n if dot(n,wo)>0 else -n)
        # brdf = disney_brdf(material, onb.normal, wo, w_i, onb.binormal, onb.tangent)
        # pdf = t3.z / pi
        # sample = struct(pdf=pdf, w_i=w_i, brdf=brdf)
        ray = make_ray(p, sample.w_i, 1e-4, 1e30)

        pdf_bsdf = sample.pdf
        if pdf_bsdf < 1e-4:
            break
        beta *= sample.brdf * abs(dot(sample.w_i, n)) / pdf_bsdf

        # rr
        l = dot(make_float3(0.212671, 0.715160, 0.072169), beta)
        if l == 0.0:
            break
        if depth >= rr_depth and l < 1.0:
            q = max(l, 0.05)
            r = sampler.next()
            if r >= q:
                break
            beta *= 1.0 / q
    if any(isnan(radiance)):
        radiance = make_float3(0.0)
    accum = accum_image.read(coord).xyz
    accum_image.write(coord, make_float4(accum + clamp(radiance, 0.0, 30.0), 1.0))



@luisa.func
def linear_to_srgb(x: float3):
    return clamp(select(1.055 * x ** (1.0 / 2.4) - 0.055,
                12.92 * x,
                x <= 0.00031308),
                0.0, 1.0)

@luisa.func
def hdr2ldr_kernel(hdr_image, ldr_image, scale: float):
    coord = dispatch_id().xy
    hdr = hdr_image.read(coord)
    ldr = linear_to_srgb(hdr.xyz * scale)
    ldr_image.write(coord, make_float4(ldr, 1.0))



luisa.log_level_info()

accum_image = luisa.Texture2D.zeros(*resolution, 4, float)
ldr_image = luisa.Texture2D.empty(*resolution, 4, float)

# compute & display the progressively converging image in a window

path_tracer(accum_image, accel, make_int2(*resolution), 1234567, dispatch_size=resolution)
luisa.synchronize()

t0 = perf_counter()
for i in range(1024):
    path_tracer(accum_image, accel, make_int2(*resolution), i, dispatch_size=resolution)
luisa.synchronize()
t1 = perf_counter()

hdr2ldr_kernel(accum_image, ldr_image, 1/1025, dispatch_size=[*resolution, 1])
# save image when window is closed
Image.fromarray(ldr_image.to('byte').numpy()).save(outfile + str(max_depth) + "_" + "{:.2f}".format(t1 - t0) + ".png")
print("1024 spp time:", t1-t0)


# gui = luisa.GUI("Cornell Box", resolution=resolution)
# frame_id = 0
# while gui.running():
#     path_tracer(accum_image, accel, make_int2(*resolution), frame_id, dispatch_size=resolution)
#     frame_id += 1
#     if frame_id % 1 == 0:
#         hdr2ldr_kernel(accum_image, ldr_image, 1/frame_id, dispatch_size=[*resolution, 1])
#         gui.set_image(ldr_image)
#         gui.show()