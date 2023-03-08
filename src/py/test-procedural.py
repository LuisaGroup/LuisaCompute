from luisa import *
from luisa.types import *
from luisa.builtin import *
import numpy as np
import math
init()

state = 0
pi = math.pi


@func
def radians(deg):
    return deg * pi / 180.


def lcg():
    global state
    lcg_a = 1664525
    lcg_c = 1013904223
    state = lcg_a * state + lcg_c
    return float(state & 0x00ffffff) * (1. / float(0x01000000))


AABB = StructType(
    min=ArrayType(3, float),
    max=ArrayType(3, float)
)


@func
def _get_min(self):
    return float3(self.min[0], self.min[1], self.min[2])


@func
def _get_max(self):
    return float3(self.max[0], self.max[1], self.max[2])


AABB.add_method(_get_min, "get_min")
AABB.add_method(_get_max, "get_max")

res = 1280, 720
image = Texture2D(*res, 4, float, storage="BYTE")
aabb_count = 1024
radius = .2
aabbs = np.empty(aabb_count * 6, dtype=np.float32)
# initialize bounding boxes
for i in range(aabb_count):
    pos = [lcg() * 2. - 1., lcg() * 2. - 1., lcg() * 2. - 1.]
    for j in range(3):
        aabbs[i * 6 + j] = pos[j] * 10. - radius
        aabbs[i * 6 + j + 3] = pos[j] * 10. + radius
acc = Accel()
# add aabb to accel
aabb_buffer = Buffer(aabb_count, AABB)
aabb_buffer.copy_from(aabbs)
acc.add_procedural(aabb_buffer, 0, aabb_count, allow_compact=False)
# add a triangle to accel
vertices = [
    float3(-0.5, -0.5, -2.0),
    float3(0.5, -0.5, -1.5),
    float3(0.0, 0.5, -1.0),
]
indices = np.array([0, 1, 2], dtype=np.int32)

vertex_buffer = Buffer(3, float3)
index_buffer = Buffer(3, int)
vertex_buffer.copy_from(vertices)
index_buffer.copy_from(indices)
acc.add(vertex_buffer, index_buffer)
acc.update()


@func
def kernel(pos):
    coord = dispatch_id().xy
    size = dispatch_size().xy
    aspect = float(size.x) / float(size.y)
    p = float2(coord) / float2(size) * 2. - 1.
    fov = radians(45.8)
    ray_origin = pos
    direction = normalize(float3(p * tan(.5 * fov) * float2(aspect, 1.), -1.))
    ray = make_ray(ray_origin, direction, 1e-3, 1e3)
    q = acc.trace_all(ray, -1)
    sphere_dist = 1e3
    while q.proceed():
        if q.is_candidate_triangle():
            q.commit_triangle()
        else:
            h = q.procedural_candidate()
            aabb = aabb_buffer.read(h.prim)
            origin = (aabb.get_min() + aabb.get_max()) * .5
            L = origin - ray_origin
            cos_theta = dot(direction, normalize(L))
            if cos_theta > 0.:
                d_oc = length(L)
                tc = d_oc * cos_theta
                d = sqrt(d_oc * d_oc - tc * tc)
                if d <= radius:
                    t1c = sqrt(radius * radius - d * d)
                    dist = tc - t1c
                    if dist <= sphere_dist:
                        sphere_dist = dist
                        normal = normalize(
                            ray_origin + direction * dist - origin)
                        sphere_color = normal * .5 + .5
                    q.commit_procedural(dist)
    hit = q.committed_hit()
    if hit.hit_procedural():
        image.write(coord, float4(sphere_color, 1.))
    elif hit.hit_triangle():
        image.write(coord, float4(hit.bary, 0., 1.))
    else:
        image.write(coord, make_float4(0, 0, 0, 1.))


gui = GUI("Test ray tracing", res)
pos = float3(0., 0., 18.0)
while gui.running():
    kernel(pos, dispatch_size=(*res, 1))
    gui.set_image(image)
    gui.show()
synchronize()
