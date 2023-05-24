from .dylibs import lcapi
from . import globalvars
from .globalvars import get_global_device
from .mathtypes import *
from .func import func
from .types import to_lctype, BuiltinFuncBuilder, uint
from .builtin import bitwise_cast, check_exact_signature
from .hit import TriangleHit
from .rayquery import rayQueryAllType, rayQueryAnyType, Ray


@func
def make_ray(origin: float3, direction: float3, t_min: float, t_max: float):
    r = Ray()
    r._origin[0] = origin[0]
    r._origin[1] = origin[1]
    r._origin[2] = origin[2]
    r._dir[0] = direction[0]
    r._dir[1] = direction[1]
    r._dir[2] = direction[2]
    r.t_min = t_min
    r.t_max = t_max
    return r


@func
def inf_ray(origin: float3, direction: float3):
    r = Ray()
    r._origin[0] = origin[0]
    r._origin[1] = origin[1]
    r._origin[2] = origin[2]
    r._dir[0] = direction[0]
    r._dir[1] = direction[1]
    r._dir[2] = direction[2]
    r.t_min = 0
    r.t_max = 1e38
    return r


@func
def offset_ray_origin(p: float3, n: float3):
    origin = 1 / 32
    float_scale = 1.0 / 65536.0
    int_scale = 256.0
    of_i = int3(int_scale * n)
    int_p = int3()
    int_p.x = bitwise_cast(int, p.x)
    int_p.y = bitwise_cast(int, p.y)
    int_p.z = bitwise_cast(int, p.z)
    p_i_tmp = int_p + select(of_i, -of_i, p < 0.0)
    p_i = float3()
    p_i.x = bitwise_cast(float, p_i_tmp.x)
    p_i.y = bitwise_cast(float, p_i_tmp.y)
    p_i.z = bitwise_cast(float, p_i_tmp.z)
    return select(p_i, p + float_scale * n, abs(p) < origin)


class Accel:
    def __init__(self, hint: lcapi.AccelUsageHint = lcapi.AccelUsageHint.FAST_BUILD, allow_compact: bool = False,
                 allow_update: bool = False):
        self._accel = get_global_device().create_accel(hint, allow_compact, allow_update)
        self.handle = self._accel.handle()

    @staticmethod
    def empty():
        return Accel()

    def add(self, vertex_buffer, triangle_buffer, transform=float4x4(1), allow_compact: bool = True,
            allow_update: bool = False, visibility_mask: int = -1, opaque: bool = True):
        self._accel.emplace_back(vertex_buffer.handle, 0, vertex_buffer.bytesize, to_lctype(vertex_buffer.dtype).size(),
                                 triangle_buffer.handle, 0, triangle_buffer.bytesize, transform, allow_compact,
                                 allow_update, visibility_mask, opaque)

    def add_procedural(self, aabb_buffer, aabb_start_index: int = 0, aabb_count=None, transform=float4x4(1),
                       allow_compact: bool = True, allow_update: bool = False, visibility_mask: int = -1,
                       opaque: bool = True):
        assert (aabb_buffer.stride == 24)
        var_aabb_count = None
        if aabb_count is None:
            var_aabb_count = aabb_buffer.size
        else:
            var_aabb_count = aabb_count
        self._accel.emplace_procedural(aabb_buffer.handle, aabb_start_index, var_aabb_count, transform, allow_compact,
                                       allow_update, visibility_mask, opaque)

    def set(self, index, vertex_buffer, triangle_buffer, transform=float4x4(1), allow_compact: bool = True,
            allow_update: bool = False, visibility_mask: int = -1, opaque: bool = True):
        self._accel.set(index, vertex_buffer.handle, 0, vertex_buffer.bytesize, to_lctype(vertex_buffer.dtype).size(),
                        triangle_buffer.handle, 0, triangle_buffer.bytesize, transform, allow_compact, allow_update,
                        visibility_mask, opaque)

    def set_procedural(self, index: int, aabb_buffer, aabb_start_index: int = 0, aabb_count=None, transform=float4x4(1),
                       allow_compact: bool = True, allow_update: bool = False, visibility_mask: int = -1,
                       opaque: bool = True):
        assert (aabb_buffer.stride == 24)
        var_aabb_count = None
        if aabb_count is None:
            var_aabb_count = aabb_buffer.size
        else:
            var_aabb_count = aabb_count
        self._accel.set_procedural(index, aabb_buffer.handle, aabb_start_index, var_aabb_count, transform,
                                   allow_compact, allow_update, visibility_mask, opaque)

    def add_buffer_view(self, vertex_buffer, vertex_byteoffset, vertex_bytesize, vertex_stride, triangle_buffer,
                        triangle_byteoffset, triangle_bytesize, transform=float4x4(1), allow_compact: bool = True,
                        allow_update: bool = False, visibility_mask: int = -1, opaque: bool = True):
        assert (triangle_byteoffset & 15) == 0 and (vertex_byteoffset & 15) == 0
        assert vertex_byteoffset + vertex_bytesize <= vertex_buffer.bytesize
        assert triangle_byteoffset + triangle_bytesize <= triangle_buffer.bytesize
        self._accel.emplace_back(vertex_buffer.handle, vertex_byteoffset, vertex_bytesize, vertex_stride,
                                 triangle_buffer.handle, triangle_byteoffset, triangle_bytesize, transform,
                                 allow_compact, allow_update, visibility_mask, opaque)

    def set_buffer_view(self, index, vertex_buffer, vertex_byteoffset, vertex_bytesize, vertex_stride, triangle_buffer,
                        triangle_byteoffset, triangle_bytesize, transform=float4x4(1), allow_compact: bool = True,
                        allow_update: bool = False, visibility_mask: int = -1, opaque: bool = True):
        assert vertex_byteoffset + vertex_bytesize <= vertex_buffer.bytesize
        assert triangle_byteoffset + triangle_bytesize <= triangle_buffer.bytesize
        self._accel.set(index, vertex_buffer.handle, vertex_byteoffset, vertex_bytesize, vertex_stride,
                        triangle_buffer.handle, triangle_byteoffset, triangle_bytesize, transform, allow_compact,
                        allow_update, visibility_mask, opaque)

    def pop(self):
        self._accel.pop_back()

    def __len__(self):
        return self._accel.size()

    def set_transform_on_update(self, index, transform: float4x4):
        self._accel.set_transform_on_update(index, transform)

    def set_visibility_on_update(self, index, visibility_mask: int):
        self._accel.set_visibility_on_update(index, visibility_mask)

    def update(self, sync=False, stream=None):
        if stream is None:
            stream = globalvars.vars.stream
        stream.update_accel(self._accel)
        if sync:
            stream.synchronize()

    def update_instance_buffer(self, sync=False, stream=None):
        if stream is None:
            stream = globalvars.vars.stream
        stream.update_instance_buffer(self._accel)
        if sync:
            stream.synchronize()

    @BuiltinFuncBuilder
    def trace_closest(self, ray, vis_mask):
        check_exact_signature([Ray, uint], [ray, vis_mask], "trace_closest")
        expr = lcapi.builder().call(to_lctype(TriangleHit), lcapi.CallOp.RAY_TRACING_TRACE_CLOSEST, [self.expr, ray.expr, vis_mask.expr])
        return TriangleHit, expr

    @BuiltinFuncBuilder
    def trace_any(self, ray, vis_mask):
        check_exact_signature([Ray, uint], [ray, vis_mask], "trace_any")
        expr = lcapi.builder().call(to_lctype(bool), lcapi.CallOp.RAY_TRACING_TRACE_ANY, [self.expr, ray.expr, vis_mask.expr])
        return bool, expr

    @BuiltinFuncBuilder
    def instance_transform(self, index):
        check_exact_signature([uint], [index], "instance_transform")
        expr = lcapi.builder().call(to_lctype(float4x4), lcapi.CallOp.RAY_TRACING_INSTANCE_TRANSFORM, [self.expr, index.expr])
        return float4x4, expr

    @BuiltinFuncBuilder
    def set_instance_transform(self, index, transform):
        check_exact_signature([uint, float4x4], [index, transform], "set_instance_transform")
        expr = lcapi.builder().call(to_lctype(float4x4), lcapi.CallOp.RAY_TRACING_SET_INSTANCE_TRANSFORM, [self.expr, index.expr, transform.expr])
        return float4x4, expr


    @BuiltinFuncBuilder
    def set_instance_visibility(self, index, visibility_mask):
        check_exact_signature([uint, uint], [index, visibility_mask], "set_instance_visibility")
        expr = lcapi.builder().call(lcapi.CallOp.RAY_TRACING_SET_INSTANCE_VISIBILITY, [self.expr, index.expr, visibility_mask.expr])
        return None, expr

    @BuiltinFuncBuilder
    def set_instance_opacity(self, index, opacity):
        check_exact_signature([uint, bool], [index, opacity], "set_instance_opacity")
        expr = lcapi.builder().call(lcapi.CallOp.RAY_TRACING_SET_INSTANCE_OPACITY, [self.expr, index.expr, opacity.expr])
        return None, expr

    @BuiltinFuncBuilder
    def query_all(self, ray, vis_mask):
        check_exact_signature([Ray, uint], [ray, vis_mask], "query_all")
        expr = lcapi.builder().call(to_lctype(rayQueryAllType), lcapi.CallOp.RAY_TRACING_QUERY_ALL, [self.expr, ray.expr, vis_mask.expr])
        return rayQueryAllType, expr

    @BuiltinFuncBuilder
    def query_any(self, ray, vis_mask: int):
        check_exact_signature([Ray, uint], [ray, vis_mask], "query_any")
        expr = lcapi.builder().call(to_lctype(rayQueryAnyType), lcapi.CallOp.RAY_TRACING_QUERY_ANY, [self.expr, ray.expr, vis_mask.expr])
        return rayQueryAllType, expr
