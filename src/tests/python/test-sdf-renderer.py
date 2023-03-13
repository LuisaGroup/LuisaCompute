from luisa import *
from luisa.builtin import *
from luisa.types import *
from luisa.util import *
import math
init()
max_ray_depth = 6
eps = 1e-4
inf = 1e10
fov = 0.23
dist_limit = 100.0
camera_pos = float3(0.0, 0.32, 3.7)
light_pos = float3(-1.5, 0.6, 0.3)
light_normal = float3(1.0, 0.0, 0.0)
light_radius = 2.0
State = StructType(value=uint)


@func
def intersect_light(pos: float3, d: float3):
    cos_w = dot(-d, light_normal)
    dist = dot(d, light_pos - pos)
    D = dist / cos_w
    dist_to_center = distance_squared(light_pos, pos + D * d)
    valid = cos_w > 0.0 and dist > 0.0 and dist_to_center < light_radius * light_radius
    return ite(valid, D, inf)


@func
def tea(v0, v1):
    s0 = uint()
    for i in range(4):
        s0 += 0x9e3779b9
        v0 += ((v1 << 4) + 0xa341316c) ^ (v1 + s0) ^ ((v1 >> 5) + 0xc8013ea4)
        v1 += ((v0 << 4) + 0xad90777d) ^ (v0 + s0) ^ ((v0 >> 5) + 0x7e95761e)
    return v0


@func
def rand(state):
    lcg_a = 1664525
    lcg_c = 1013904223
    state.value = lcg_a * state.value + lcg_c
    return float(state.value & 0x00ffffff) * (1. / float(0x01000000))


@func
def out_dir(n, seed):
    u = ite(abs(n.y) < (1.0 - eps), normalize(cross(n, float3(0., 1., 0.))), float3(1., 0., 0.))
    v = cross(n, u)
    phi = 2.0 * math.pi * rand(seed)
    ay = sqrt(rand(seed))
    ax = sqrt(1.0 - ay * ay)
    return ax * (cos(phi) * u + sin(phi) * v) + ay * n


@func
def make_nested(f):
    freq = 40.0
    f *= freq
    f = ite(f < 0., ite(int(f) % 2 == 0, 1. - fract(f), fract(f)), f)
    return (f - 0.2) * (1.0 / freq)


@func
def sdf(o):
    wall = min(o.y + 0.1, o.z + 0.4)
    sphere = distance(o, float3(0.0, 0.35, 0.0)) - 0.36
    q = abs(o - float3(0.8, 0.3, 0.0)) - 0.3
    box = length(max(q, 0.0)) + min(max(max(q.x, q.y), q.z), 0.0)
    O = o - float3(-0.8, 0.3, 0.0)
    d = float2(length(float2(O.x, O.z)) - 0.3, abs(O.y) - 0.3)
    cylinder = min(max(d.x, d.y), 0.0) + length(max(d, 0.0))
    geometry = make_nested(min(min(sphere, box), cylinder))
    g = max(geometry, -(0.32 - (o.y * 0.6 + o.z * 0.8)))
    return min(wall, g)


@func
def ray_march(p, d):
    dist = 0.
    for j in range(100):
        s = sdf(p + dist * d)
        if s <= 1e-6 or dist >= inf:
            break
        dist += s
    return min(dist, inf)


@func
def sdf_normal(p):
    d = 1e-3
    n = float3()
    sdf_center = sdf(p)
    inc = float3()
    for i in range(3):
        inc = p
        inc[i] += d
        n[i] = (1.0 / d) * (sdf(inc) - sdf_center)
    return normalize(n)


NextHit = StructType(
    closest=float,
    normal=float3,
    c=float3
)


@func
def next_hit(data: NextHit, pos, d):
    data.closest = inf
    data.normal = float3()
    data.c = float3()
    ray_march_dist = ray_march(pos, d)
    if (ray_march_dist < min(dist_limit, data.closest)):
        data.closest = ray_march_dist
        hit_pos = pos + d * data.closest
        data.normal = sdf_normal(hit_pos)
        t = int((hit_pos.x + 10.0) * 1.1 + 0.5) % 3
        data.c = float3(0.4) + float3(0.3, 0.2, 0.3) * \
            ite(t == int3(0, 1, 2), float3(1.0), float3(0.0))


@func
def render_kernel(seed_image, accum_image, frame_index):
    set_block_size(16, 8, 1)
    resolution = float2(dispatch_size().xy)
    coord = dispatch_id().xy
    if (frame_index == 0):
        seed_image.write(coord, tea(coord.x, coord.y))

    aspect_ratio = resolution.x / resolution.y
    pos = camera_pos
    seed = State()
    seed.value = seed_image.read(coord)
    ux = rand(seed)
    uy = rand(seed)
    uv = float2(dispatch_id().x + ux, dispatch_size().y -
                1 - dispatch_id().y + uy)
    d = float3(2.0 * fov * uv / resolution.y - fov *
               float2(aspect_ratio, 1.0) - 1e-5, -1.0)
    d = normalize(d)
    throughput = float3(1)
    hit_light = 0.0
    accum_color = float3()
    for depth in range(max_ray_depth):
        data = NextHit()
        next_hit(data, pos, d)
        dist_to_light = intersect_light(pos, d)
        if (dist_to_light < data.closest):
            hit_light = 1.0
            break
        if (length_squared(data.normal) == 0.0):
            break
        hit_pos = pos + data.closest * d
        d = out_dir(data.normal, seed)
        pos = hit_pos + 1e-4 * d
        throughput *= data.c
    accum_color = lerp(accum_image.read(coord).xyz, throughput.xyz * hit_light, 1.0 / (frame_index + 1.0))
    accum_image.write(coord, float4(accum_color, 1.0))
    seed_image.write(coord, seed.value)


@func
def linear_to_srgb(x: float3):
    return clamp(select(1.055 * x ** (1.0 / 2.4) - 0.055,
                        12.92 * x,
                        x <= 0.00031308),
                 0.0, 1.0)


@func
def hdr2ldr_kernel(hdr_image, ldr_image, scale: float):
    coord = dispatch_id().xy
    hdr = hdr_image.read(coord)
    ldr = linear_to_srgb(hdr.xyz * scale)
    ldr_image.write(coord, float4(ldr, 1.0))

res = (1280, 720)
seed_image = Texture2D(*res, 1, uint, storage="INT")
accum_image=Texture2D(*res, 4, float, storage="FLOAT")
ldr_image=Texture2D(*res, 4, float, storage="BYTE")
gui = GUI("Test cornell box", res)
frame = 0
while gui.running():
    render_kernel(seed_image, accum_image, frame, dispatch_size=(*res, 1))
    frame += 1
    hdr2ldr_kernel(accum_image, ldr_image, 2.0, dispatch_size=(*res, 1))
    gui.set_image(ldr_image)
    gui.show()
synchronize()