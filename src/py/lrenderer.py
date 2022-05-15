import luisa
from luisa.mathtypes import *
from math import pi
from PIL import Image
from luisa.util import RandomSampler

from disney import *

from cbox_disney import models # (filename, mat, [emission], [transform])
from cbox_disney import const_env_light, camera_pos, camera_dir, camera_up, camera_fov
from cbox_disney import resolution, max_depth, rr_depth
from parseobj import parseobj

if camera_fov > pi: # likely to be in degrees; convert to radian
    camera_fov *= pi / 180


# load scene

meshes = [] # each mesh
heapindex = {} # idx -> buffer
emission = [] # emission of each mesh
light_meshid = [] # list of emissive meshes
materials = []
tricount = [] # number of triangles in each mesh

def flatten_list(a):
    s = []
    for x in a:
        s += x
    return s

luisa.init()
for idx, model in enumerate(models):
    filename, material = model[0:2]
    materials.append(material)
    print("loading", filename)
    v,vn,f = parseobj(open(filename))
    print(v,f)
    assert len(v) > 0
    # assert len(v) == len(vn) > 0
    # add mesh
    vertex_buffer = luisa.buffer(list(map(lambda x: float3(*x), v)))
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
    pdf = (1.0 - env_prob) * sqr_dist / (n * n1 * area * cos_light)
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
        p0 = heap.buffer_read(float3, inst * 2 + 1, i0)
        p1 = heap.buffer_read(float3, inst * 2 + 1, i1)
        p2 = heap.buffer_read(float3, inst * 2 + 1, i2)
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
def generate_ray(p: float2): # p: [-1,1]^2
    fov = 39 / 180 * 3.1415926
    origin = make_float3(278, 273, -800)
    pixel = origin + make_float3(p * tan(0.5 * fov), 1.0)
    direction = normalize(pixel - origin)
    return make_ray(origin, direction, 0.0, 1e30) # TODO

@luisa.func
def generate_camera_ray(sampler, resolution):
    coord = dispatch_id().xy
    frame_size = float(min(resolution.x, resolution.y))
    pixel = (make_float2(coord) + sampler.next2f()) / frame_size * 2.0 - 1.0 # remapped to [-1,1] in shorter axis
    ray = generate_ray(pixel * make_float2(1.0, -1.0))
    return ray


@luisa.func
def cosine_sample_hemisphere(u: float2):
    r = sqrt(u.x)
    phi = 2.0 * 3.1415926 * u.y
    return make_float3(r * cos(phi), r * sin(phi), sqrt(1.0 - u.x))


@luisa.func
def balanced_heuristic(pdf_a, pdf_b):
    return pdf_a / max(pdf_a + pdf_b, 1e-4)


@luisa.func
def path_tracer(accum_image, accel, resolution, frame_index):
    set_block_size(8, 8, 1)
    coord = dispatch_id().xy
    sampler = RandomSampler(make_int3(coord, frame_index))
    ray = generate_camera_ray(sampler, resolution)

    radiance = make_float3(0.0)
    beta = make_float3(1.0)
    pdf_bsdf = 1e30

    # light_position = make_float3(-0.24, 1.98, 0.16)
    # light_u = make_float3(-0.24, 1.98, -0.22) - light_position
    # light_v = make_float3(0.23, 1.98, 0.16) - light_position
    # light_emission = make_float3(17.0, 12.0, 4.0)
    # light_area = length(cross(light_u, light_v))
    # light_normal = normalize(cross(light_u, light_v))

    for depth in range(max_depth):
        # trace
        hit = accel.trace_closest(ray)
        # miss: evaluate environment light
        if hit.miss():
            # ============== DEBUG ==================
            accum_image.write(coord, make_float4(0.0, 0.0, 0.0, 1.0))
            return

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
        p0 = heap.buffer_read(float3, hit.inst * 2 + 1, i0)
        p1 = heap.buffer_read(float3, hit.inst * 2 + 1, i1)
        p2 = heap.buffer_read(float3, hit.inst * 2 + 1, i2)
        # get hit position, onb and albedo (surface color)
        p = hit.interpolate(p0, p1, p2)
        n = normalize(cross(p1 - p0, p2 - p0))


        # ============== DEBUG ==============
        radiance = n * float3(0.5) + float3(0.5)
        # radiance = float3(hit.inst / 10 + 0.1)
        accum_image.write(coord, make_float4(radiance * float(frame_index + 1), 1.0))
        return


        # TODO: incorporate interpolated shading normal
        onb = make_onb(n)
        # black if hit backside
        wo = -ray.get_dir()
        cos_wo = dot(wo, n)
        if cos_wo < 1e-4:
            break
        material = material_buffer.read(hit.inst)
        emission = emission_buffer.read(hit.inst)

        # hit light
        if any(emission != float3(0)):
            if depth == 0:
                radiance += emission
            else:
                pdf_light = mesh_light_sampled_pdf(p, ray.get_origin(), hit.inst, p0, p1, p2)
                mis_weight = balanced_heuristic(pdf_bsdf, pdf_light)
                radiance += mis_weight * beta * emission
            break

        # sample light
        light = sample_light(p, sampler) # (wi, dist, pdf, eval)
        shadow_ray = make_ray(offset_ray_origin(p, n), light.wi, 0.0, light.dist)
        occluded = accel.trace_any(shadow_ray)
        cos_wi_light = dot(light.wi, n)
        if not occluded and cos_wi_light > 1e-4:
            bsdf = disney_brdf(material, onb.normal, wo, light.wi, onb.binormal, onb.tangent)
            pdf_bsdf = disney_pdf(material, onb.normal, wo, light.wi, onb.binormal, onb.tangent)
            mis_weight = balanced_heuristic(light.pdf, pdf_bsdf)
            radiance += beta * bsdf * cos_wi_light * mis_weight * light.eval / max(light.pdf, 1e-4)

        # sample BSDF (pdf, w_i, brdf)
        sampled = sample_disney_brdf(material, onb.normal, wo, onb.binormal, onb.tangent, sampler)
        new_direction = sampled.w_i
        ray = make_ray(offset_ray_origin(p, n), new_direction, 0.0, 1e30)
        pdf_bsdf = sampled.pdf
        beta *= sampled.brdf * dot(sampled.w_i, n) / pdf_bsdf

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



luisa.lcapi.log_level_info()

res = 1024, 1024
accum_image = luisa.Texture2D.zeros(*res, 4, float)
ldr_image = luisa.Texture2D.empty(*res, 4, float)

# compute & display the progressively converging image in a window
gui = luisa.GUI("Cornell Box", resolution=res)
frame_id = 0

while gui.running():
    path_tracer(accum_image, accel, make_int2(*res), frame_id, dispatch_size=res)
    frame_id += 1
    if frame_id % 16 == 0:
        hdr2ldr_kernel(accum_image, ldr_image, 1/frame_id, dispatch_size=[*res, 1])
        gui.set_image(ldr_image)
        gui.show()

# save image when window is closed
# Image.fromarray(final_image.to('byte').numpy()).save("cornell.png")

