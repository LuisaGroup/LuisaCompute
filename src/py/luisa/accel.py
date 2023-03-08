import lcapi
from . import globalvars
from .globalvars import get_global_device
from .struct import StructType
from .array import ArrayType
from .mathtypes import *
from .func import func
from .types import to_lctype
from .builtin import _builtin_call, bitwise_cast
from .hit import Hit
from .rayquery import rayQueryType, rayQuery
# Ray
Ray = StructType(16, _origin=ArrayType(3,float), t_min=float, _dir=ArrayType(3,float), t_max=float)

@func
def make_ray(origin: float3, direction: float3, t_min: float, t_max:float):
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

@func
def get_origin(self):
    return float3(self._origin[0], self._origin[1], self._origin[2])
Ray.add_method(get_origin)

@func
def get_dir(self):
    return float3(self._dir[0], self._dir[1], self._dir[2])
Ray.add_method(get_dir)

@func
def set_origin(self, val: float3):
    self._origin[0] = val.x
    self._origin[1] = val.y
    self._origin[2] = val.z
Ray.add_method(set_origin)

@func
def set_dir(self, val: float3):
    self._dir[0] = val.x
    self._dir[1] = val.y
    self._dir[2] = val.z
Ray.add_method(set_dir)
# Var<float> interpolate(Expr<Hit> hit, Expr<float> a, Expr<float> b, Expr<float> c) noexcept {
#     return (1.0f - hit.bary.x - hit.bary.y) * a + hit.bary.x * b + hit.bary.y * c;
# }

# Var<float2> interpolate(Expr<Hit> hit, Expr<float2> a, Expr<float2> b, Expr<float2> c) noexcept {
#     return (1.0f - hit.bary.x - hit.bary.y) * a + hit.bary.x * b + hit.bary.y * c;
# }

# Var<float3> interpolate(Expr<Hit> hit, Expr<float3> a, Expr<float3> b, Expr<float3> c) noexcept {
#     return (1.0f - hit.bary.x - hit.bary.y) * a + hit.bary.x * b + hit.bary.y * c;
# }


class Accel:
    def __init__(self,  hint:lcapi.AccelUsageHint = lcapi.AccelUsageHint.FAST_BUILD, allow_compact:bool = False, allow_update:bool = False):
        self._accel = get_global_device().create_accel(hint, allow_compact, allow_update)
        self.handle = self._accel.handle() 

    @staticmethod
    def empty():
        return Accel()

    def add(self, vertex_buffer, triangle_buffer, transform = float4x4(1), allow_compact:bool = True, allow_update:bool = False, visibility_mask:int=-1, opaque:bool = True):
        self._accel.emplace_back(vertex_buffer.handle, 0, vertex_buffer.bytesize, to_lctype(vertex_buffer.dtype).size(), triangle_buffer.handle, 0, triangle_buffer.bytesize, transform, allow_compact, allow_update, visibility_mask, opaque)
    def add_procedural(self, aabb_buffer, aabb_start_index:int = 0, aabb_count = None, transform = float4x4(1), allow_compact:bool = True, allow_update:bool = False, visibility_mask:int=-1, opaque:bool = True):
        assert(aabb_buffer.stride == 24)
        var_aabb_count = None
        if aabb_count == None:
            var_aabb_count = aabb_buffer.size
        else:
            var_aabb_count = aabb_count
        self._accel.emplace_procedural(aabb_buffer.handle, aabb_start_index, var_aabb_count, transform, allow_compact, allow_update, visibility_mask, opaque)
    def set(self, index, vertex_buffer, triangle_buffer, transform = float4x4(1),allow_compact:bool = True, allow_update:bool = False, visibility_mask:int = -1, opaque:bool = True):
        self._accel.set(index, vertex_buffer.handle, 0, vertex_buffer.bytesize, to_lctype(vertex_buffer.dtype).size(), triangle_buffer.handle, 0, triangle_buffer.bytesize, transform, allow_compact, allow_update, visibility_mask, opaque)
    def set_procedural(self, index: int, aabb_buffer, aabb_start_index:int = 0, aabb_count = None, transform = float4x4(1), allow_compact:bool = True, allow_update:bool = False, visibility_mask:int=-1, opaque:bool = True):
        assert(aabb_buffer.stride == 24)
        var_aabb_count = None
        if aabb_count == None:
            var_aabb_count = aabb_buffer.size
        else:
            var_aabb_count = aabb_count
        self._accel.set_procedural(index, aabb_buffer.handle, aabb_start_index, var_aabb_count, transform, allow_compact, allow_update, visibility_mask, opaque)

    def add_buffer_view(self, vertex_buffer, vertex_byteoffset, vertex_bytesize, vertex_stride, triangle_buffer, triangle_byteoffset, triangle_bytesize, transform = float4x4(1), allow_compact:bool = True, allow_update:bool = False, visibility_mask:int=-1, opaque:bool = True):
        assert (triangle_byteoffset & 15) == 0 and (vertex_byteoffset & 15) == 0
        assert vertex_byteoffset + vertex_bytesize <= vertex_buffer.bytesize
        assert triangle_byteoffset + triangle_bytesize <= triangle_buffer.bytesize
        self._accel.emplace_back(vertex_buffer.handle, vertex_byteoffset, vertex_bytesize, vertex_stride, triangle_buffer.handle, triangle_byteoffset, triangle_bytesize, transform, allow_compact, allow_update, visibility_mask, opaque)
    def set_buffer_view(self, index, vertex_buffer, vertex_byteoffset, vertex_bytesize, vertex_stride, triangle_buffer, triangle_byteoffset, triangle_bytesize, transform = float4x4(1),allow_compact:bool = True, allow_update:bool = False, visibility_mask:int = -1, opaque:bool = True):
        assert vertex_byteoffset + vertex_bytesize <= vertex_buffer.bytesize
        assert triangle_byteoffset + triangle_bytesize <= triangle_buffer.bytesize
        self._accel.set(index, vertex_buffer.handle, vertex_byteoffset, vertex_bytesize, vertex_stride, triangle_buffer.handle, triangle_byteoffset, triangle_bytesize, transform, allow_compact, allow_update, visibility_mask, opaque)
    def pop(self):
        self._accel.pop_back()

    def __len__(self):
        return self._accel.size()

    def set_transform_on_update(self, index, transform: float4x4):
        self._accel.set_transform_on_update(index, transform)

    def set_visibility_on_update(self, index, visibility_mask: int):
        self._accel.set_visibility_on_update(index, visibility_mask)

    def update(self, sync = False, stream = None):
        if stream is None:
            stream = globalvars.stream
        stream.update_accel(self._accel)
        if sync:
            stream.synchronize()

    @func
    def trace_closest(self, ray: Ray, vis_mask: int):
        return _builtin_call(Hit, "RAY_TRACING_TRACE_CLOSEST", self, ray, vis_mask)

    @func
    def trace_any(self, ray: Ray, vis_mask: int):
        return _builtin_call(bool, "RAY_TRACING_TRACE_ANY", self, ray, vis_mask)

    @func
    def instance_transform(self, index: int):
        return _builtin_call(float4x4, "RAY_TRACING_INSTANCE_TRANSFORM", self, index)

    @func
    def set_instance_transform(self, index: int, transform: float4x4):
        _builtin_call("RAY_TRACING_SET_INSTANCE_TRANSFORM", self, index, transform)

    @func
    def set_instance_visibility(self, index: int, visibility_mask: int):
        _builtin_call("RAY_TRACING_SET_INSTANCE_VISIBILITY", self, index, visibility_mask)
    @func
    def set_instance_visibility(self, index: int, visibility_mask: int):
        _builtin_call("RAY_TRACING_SET_INSTANCE_OPACITY", self, index, visibility_mask)
    @func
    def trace_all(self, ray: Ray, vis_mask: int):
        return _builtin_call(rayQueryType, "RAY_TRACING_TRACE_ALL", self, ray, vis_mask)