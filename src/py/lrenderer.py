import luisa
from luisa.mathtypes import *
from math import pi
from PIL import Image
from luisa.util import RandomSampler

from disney import *




# position of each vertex
vertices = list(map(lambda x:float3(*x), [[-1,0,1], [1,0,1], [1,0,-1], [-1,0,-1], [-1,2,1], [-1,2,-1], [1,2,-1], [1,2,1], # room
    [.53,.6,.75], [.7,.6,.17], [.13,.6,0], [-.05,.6,.57], [.53,0,.75], [.7,0,.17], [.13,0,0], [-.05,0,.57], # short cube
    [-.53,1.2,.09], [.04,1.2,-.09], [-.14,1.2,-.67], [-.71,1.2,-.49], [-.53,0,.09], [.04,0,-.09], [-.14,0,-.67], [-.71,0,-.49], # tall cube
    [-.24,1.98,.16], [-.24,1.98,-.22], [.23,1.98,-.22], [.23,1.98,.16]])) # light
# vertex indices of triangles in each mesh
meshes = [[0,1,2,0,2,3], [4,5,6,4,6,7], [3,2,6,3,6,5], [2,1,7,2,7,6], [0,3,5,0,5,4], # room
         [8,9,10,8,10,11,15,11,10,15,10,14,12,8,11,12,11,15,13,9,8,13,8,12,14,10,9,14,9,13,12,13,14,12,14,15], # short cube
         [16,17,18,16,18,19,20,16,19,20,19,23,23,19,18,23,18,22,22,18,17,22,17,21,21,17,16,21,16,20,20,21,22,20,22,23], # tall cube
         [24,25,26,24,26,27]] # light


rr_depth = 3

white = DisneyMaterial.default.copy()
white.base_color = float3(0.725, 0.71, 0.68)
red = DisneyMaterial.default.copy()
red.base_color = float3(0.63, 0.065, 0.05)
green = DisneyMaterial.default.copy()
green.base_color = float3(0.14, 0.45, 0.091)
black = DisneyMaterial.default.copy()
black.base_color = float3(0)


# luisa.init()

# @luisa.func
# def test():
#     sampler = RandomSampler(dispatch_id())
#     normal = float3(0,0,1)
#     binormal = normalize(select(
#             make_float3(0.0, -normal.z, normal.y),
#             make_float3(-normal.y, normal.x, 0.0),
#             abs(normal.x) > abs(normal.z)))
#     tangent = normalize(cross(binormal, normal))
#     w_o = normalize(float3(0,1,1))
#     res = sample_disney_brdf(white, normal, w_o, binormal, tangent, sampler)
#     print(res.pdf, res.w_i, res.brdf)

# test(dispatch_size=1)
# quit()

materials = [white, white, white, green, red, white, white, black]

# copy scene info to device
luisa.init()
vertex_buffer, material_buffer = luisa.buffer(vertices), luisa.buffer(materials)
triangle_buffers = list(map(luisa.buffer, meshes))
triangle_buffer_table = luisa.bindless_array({i:buf for i,buf in enumerate(triangle_buffers)})
res = 1024, 1024
accel = luisa.accel([luisa.Mesh(vertex_buffer, t) for t in triangle_buffers])



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
def path_tracer(accum_image, accel, resolution, frame_index):
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
        i0 = triangle_buffer_table.buffer_read(int, hit.inst, hit.prim * 3 + 0)
        i1 = triangle_buffer_table.buffer_read(int, hit.inst, hit.prim * 3 + 1)
        i2 = triangle_buffer_table.buffer_read(int, hit.inst, hit.prim * 3 + 2)
        p0 = vertex_buffer.read(i0)
        p1 = vertex_buffer.read(i1)
        p2 = vertex_buffer.read(i2)
        p = hit.interpolate(p0, p1, p2)
        n = normalize(cross(p1 - p0, p2 - p0))
        onb = make_onb(n)
        wo = -ray.get_dir()
        cos_wo = dot(wo, n)
        if cos_wo < 1e-4:
            break
        material = material_buffer.read(hit.inst)

        # hit light
        if hit.inst == 7:
            if depth == 0:
                radiance += light_emission
            else:
                pdf_light = length_squared(p - ray.get_origin()) / (light_area * cos_wo)
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
            bsdf = disney_brdf(material, onb.normal, wo, wi_light, onb.binormal, onb.tangent)
            pdf_bsdf = disney_pdf(material, onb.normal, wo, wi_light, onb.binormal, onb.tangent)
            mis_weight = balanced_heuristic(pdf_light, pdf_bsdf)
            radiance += beta * bsdf * cos_wi_light * mis_weight * light_emission / max(pdf_light, 1e-4)

        # sample BSDF (pdf, w_i, brdf)
        sampled = sample_disney_brdf(material, onb.normal, wo, onb.binormal, onb.tangent, sampler)
        new_direction = sampled.w_i
        ray = make_ray(pp, new_direction, 0.0, 1e30)
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



luisa.lcapi.log_level_error()

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

