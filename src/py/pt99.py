# Modern path tracing in 99 lines with Luisa

import luisa
from luisa.mathtypes import *
from math import pi
from PIL import Image
from luisa.accel import make_ray

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

white, red, green = float3(0.725, 0.71, 0.68), float3(0.63, 0.065, 0.05), float3(0.14, 0.45, 0.091)
materials = [white, white, white, green, red, white, white, white]
camera_pos = make_float3(-0.01, 0.995, 5.0)
fov = 27.8 / 180 * pi

luisa.init()
vertex_buffer = luisa.buffer(vertices)
material_buffer = luisa.buffer(materials)
triangle_buffers = list(map(luisa.buffer, meshes))
triangle_buffer_table = luisa.bindless_array({i:buf for i,buf in enumerate(triangle_buffers)})
accel = luisa.accel([luisa.Mesh(vertex_buffer, t) for t in triangle_buffers])

@luisa.func
def camera_ray(pixel):
    p = camera_pos + make_float3(pixel * tan(0.5 * fov), -1.0)
    return make_ray(camera_pos, normalize(p - camera_pos), 0.0, 1e30)

@luisa.func
def cosine_sample_hemisphere(u, N):
    bN = normalize(select(float3(0.0, -N.z, N.y), float3(-N.y, N.x, 0.0), abs(N.x) > abs(N.z)))
    tang = normalize(cross(bN, N))
    r = sqrt(u.x)
    phi = 2.0 * pi * u.y
    return r * cos(phi) * tang + r * sin(phi) * bN + sqrt(1.0 - u.x) * N

@luisa.func
def path_tracer(accum_image, frame_id, resolution):
    coord = dispatch_id().xy
    sampler = luisa.RandomSampler(int3(coord, frame_id)) # builtin RNG; the sobol sampler can be used instead to improve convergence
    pixel = 2.0 / resolution * (float2(coord) + sampler.next2f()) - 1.0
    ray = camera_ray(pixel * float2(1.0, -1.0))

    radiance = make_float3(0.0)
    beta = make_float3(1.0)

    light_emission = make_float3(17.0, 12.0, 4.0)
    light_position = float3(-0.24, 1.98, 0.16)
    light_u = float3(0, 0, -0.38)
    light_v = float3(0.47, 0, 0)
    light_area = length(cross(light_u, light_v))
    light_normal = normalize(cross(light_u, light_v))

    for depth in range(5):
        # trace returns Hit (inst: instance index; prim: triangle index; bary: barycentric coordinate (position on triangle))
        hit = accel.trace_closest(ray)
        if hit.miss():
            break
        v0_id = triangle_buffer_table.buffer_read(int, hit.inst, hit.prim * 3 + 0)
        v1_id = triangle_buffer_table.buffer_read(int, hit.inst, hit.prim * 3 + 1)
        v2_id = triangle_buffer_table.buffer_read(int, hit.inst, hit.prim * 3 + 2)
        p0 = vertex_buffer.read(v0_id)
        p1 = vertex_buffer.read(v1_id)
        p2 = vertex_buffer.read(v2_id)
        p = (1.0 - hit.bary.x - hit.bary.y) * p0 + hit.bary.x * p1 + hit.bary.y * p2 # or simpler, p = hit.interpolate(p0,p1,p2)
        n = normalize(cross(p1 - p0, p2 - p0))
        cos_wi = dot(-ray.get_dir(), n)
        if cos_wi < 1e-4:
            break
        albedo = material_buffer.read(hit.inst)

        if hit.inst == 7: # hit light
            if depth == 0: # light is directly visible
                radiance += light_emission
            break

        # sample light
        p_light = light_position + sampler.next() * light_u + sampler.next() * light_v

        # use eps to avoid self-intersection. or better, use offset_ray_origin(p,n)
        d_light = length(p - p_light)
        wi_light = normalize(p_light - p)
        occluded = accel.trace_any(make_ray(p, wi_light, 1e-4, d_light - 2e-4))
        cos_wi_light = dot(wi_light, n)
        cos_light = -dot(light_normal, wi_light)

        if not occluded and cos_wi_light > 1e-4 and cos_light > 1e-4:
            pdf_light = d_light ** 2 / (light_area * cos_light)
            bsdf = 1 / pi * albedo * cos_wi_light
            radiance += beta * bsdf * light_emission / max(pdf_light, 1e-4)

        # sample BSDF
        ray = make_ray(p, cosine_sample_hemisphere(sampler.next2f(), n), 1e-4, 1e30)
        beta *= albedo

    accum_image.write(coord, accum_image.read(coord) + float4(clamp(radiance, 0.0, 30.0), 1.0))

@luisa.func
def linear_to_srgb(hdr_image, ldr_image, scale):
    linear = hdr_image.read(dispatch_id().xy).xyz * scale
    srgb = clamp(select(1.055 * linear ** (1.0 / 2.4) - 0.055, 12.92 * linear, linear <= 0.00031308), 0.0, 1.0)
    ldr_image.write(dispatch_id().xy, float4(srgb, 1.0))

res = 1024, 1024
accum_image = luisa.Texture2D.zeros(*res, 4, float)
final_image = luisa.Texture2D.empty(*res, 4, float)

frame_id = 0
gui = luisa.GUI("Cornel Box", resolution=res)
while gui.running():
    path_tracer(accum_image, frame_id, res[0], dispatch_size=(*res, 1))
    frame_id += 1
    if frame_id % 16 == 0:
        linear_to_srgb(accum_image, final_image, 1/frame_id, dispatch_size=(*res, 1))
        gui.set_image(final_image)
        gui.show()

Image.fromarray(final_image.to(luisa.PixelStorage.BYTE4).numpy()).save("cornell.png")
