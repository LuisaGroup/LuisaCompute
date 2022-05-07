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
materials, camera_pos, light_pos = [white, white, white, green, red, white, white, white], float3(-0.01, 0.995, 5.0), float3(-0.24, 1.98, 0.16)
light_u, light_v, light_area, light_normal, light_L = float3(0, 0, -0.38), float3(0.47, 0, 0), 0.1785999983549118, float3(0, -1, 0), float3(17.0, 12.0, 4.0)

# copy scene info to device
luisa.init("metal")
vertex_buffer, material_buffer, triangle_buffers = luisa.buffer(vertices), luisa.buffer(materials), list(map(luisa.buffer, meshes))
triangle_buffer_table = luisa.bindless_array({i:buf for i,buf in enumerate(triangle_buffers)})
res, accel = (1024, 1024), luisa.accel([luisa.Mesh(vertex_buffer, t) for t in triangle_buffers])

# sample reflected direction from diffuse surface
@luisa.func
def cosine_sample_hemisphere(u, N):
    bN = normalize(select(float3(0.0, -N.z, N.y), float3(-N.y, N.x, 0.0), abs(N.x) > abs(N.z)))
    return sqrt(u.x) * (cos(2.0 * pi * u.y) * normalize(cross(bN, N)) + sin(2.0 * pi * u.y) * bN) + sqrt(1.0 - u.x) * N

@luisa.func
def path_tracer(accum_image, display_image, frame_id, resolution):
    sampler = luisa.RandomSampler(int3(dispatch_id().xy, frame_id)) # builtin RNG; the sobol sampler can be used instead to improve convergence
    # generate ray from camera
    pixel = 2.0 / resolution * (float2(dispatch_id().xy) + sampler.next2f()) - 1.0
    ray = make_ray(camera_pos, normalize(camera_pos + float3(0.247 * pixel * float2(1.0, -1.0), -1.0) - camera_pos), 0.0, 1e30)
    radiance = accum_image.read(dispatch_id().xy).xyz if frame_id > 0 else float3(0.0, 0.0, 0.0)
    beta = float3(1.0)
    for depth in range(5):
        # trace returns Hit (inst: instance index; prim: triangle index; bary: barycentric coordinate (position on triangle))
        hit = accel.trace_closest(ray)
        if hit.miss() or hit.inst == 7: # miss or hit light (which is a blackbody)
            radiance += light_L if hit.inst == 7 and depth == 0 else float3(0.0, 0.0, 0.0)
            break
        # fetch hit triangle info
        v0_id = triangle_buffer_table.buffer_read(int, hit.inst, hit.prim * 3 + 0)
        v1_id = triangle_buffer_table.buffer_read(int, hit.inst, hit.prim * 3 + 1)
        v2_id = triangle_buffer_table.buffer_read(int, hit.inst, hit.prim * 3 + 2)
        p0 = vertex_buffer.read(v0_id)
        p1 = vertex_buffer.read(v1_id)
        p2 = vertex_buffer.read(v2_id)
        p = (1.0 - hit.bary.x - hit.bary.y) * p0 + hit.bary.x * p1 + hit.bary.y * p2 # or simpler, p = hit.interpolate(p0,p1,p2)
        n = normalize(cross(p1 - p0, p2 - p0))
        if dot(-ray.get_dir(), n) < 1e-4:# hit backside
            break
        albedo = material_buffer.read(hit.inst)

        # sample light
        p_light = light_pos + sampler.next() * light_u + sampler.next() * light_v
        d_light = length(p - p_light)
        wi_light = normalize(p_light - p)
        # use eps to avoid self-intersection. or better, use offset_ray_origin(p,n)
        cos_wi_light = dot(wi_light, n)
        cos_light = -dot(light_normal, wi_light)

        # compute direct lighting
        if not accel.trace_any(make_ray(p, wi_light, 1e-4, d_light - 2e-4)) and cos_wi_light > 1e-4 and cos_light > 1e-4:
            pdf_light = d_light ** 2 / (light_area * cos_light)
            radiance += beta *(1 / pi) * albedo * cos_wi_light * light_L / max(pdf_light, 1e-4)

        # sample BSDF; continue loop to compute indirect lighting
        ray = make_ray(p, cosine_sample_hemisphere(sampler.next2f(), n), 1e-4, 1e30)

        # Russian roulette
        l = dot(float3(0.212671, 0.715160, 0.072169), beta * albedo)
        if sampler.next() >= l:
            break
        beta *= albedo / l
    accum_image.write(dispatch_id().xy, float4(radiance, 1.0))
    display_image.write(dispatch_id().xy, float4((radiance / float(frame_id + 1)) ** (1.0 / 2.2), 1.0))


# compute & display the progressively converging image in a window
gui = luisa.GUI("Cornel Box", resolution=res)
frame_id, accum_image, final_image = 0, luisa.Texture2D.zeros(*res, 4, float), luisa.Texture2D.empty(*res, 4, float)
while gui.running():
    path_tracer(accum_image, final_image, frame_id, res[0], dispatch_size=(*res, 1))
    frame_id += 1
    if frame_id % 16 == 0:
        gui.set_image(final_image), gui.show(frames_in_flight=16)

# save image when window is closed
Image.fromarray(final_image.to(luisa.PixelStorage.BYTE4).numpy()).save("cornell.png")
