import lcapi
from . import globalvars
from .globalvars import get_global_device
from .structtype import StructType
from .arraytype import ArrayType
from .mathtypes import *

from .kernel import callable_method, callable
from .types import ref
from .builtin import _builtin_call, _builtin_cast

# Ray
Ray = StructType(16, _origin=ArrayType(float,3), t_min=float, _dir=ArrayType(float,3), t_max=float)

@callable_method(Ray)
def get_origin(self: ref(Ray)):
    return float3(self._origin[0], self._origin[1], self._origin[2])

@callable_method(Ray)
def get_dir(self: ref(Ray)):
    return float3(self._dir[0], self._dir[1], self._dir[2])

@callable_method(Ray)
def set_origin(self: ref(Ray), val: float3):
    self._origin[0] = val.x
    self._origin[1] = val.y
    self._origin[2] = val.z

@callable_method(Ray)
def set_dir(self: ref(Ray), val: float3):
    self._dir[0] = val.x
    self._dir[1] = val.y
    self._dir[2] = val.z



# Hit
Hit = StructType(16, inst=int, prim=int, bary=float2)

@callable_method(Hit)
def miss(self: ref(Hit)):
    return self.inst == -1

_uHitLCtype = lcapi.Type.from_("struct<16,uint,uint,vector<float,2>>")


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
    def __init__(self):
        self._accel = get_global_device().create_accel(lcapi.AccelUsageHint.FAST_TRACE)
        self.handle = self._accel.handle()

    def add(self, mesh, transform = float4x4.identity(), visible = True):
        self._accel.emplace_back(mesh.handle, transform, visible)

    def build(self):
        globalvars.stream.add(self._accel.build_command(lcapi.AccelBuildRequest.PREFER_UPDATE))

    @callable
    def trace_closest(self: Accel, ray: Ray):
        return _builtin_cast(Hit, "BITWISE", _builtin_call(UHit, "TRACE_CLOSEST", [self, ray]))

    @callable
    def trace_any(self: Accel, ray: Ray):
        return _builtin_cast(Hit, "BITWISE", _builtin_call(UHit, "TRACE_ANY", [self, ray]))

    @callable
    def instance_transform(self: Accel, instance_id: int):
        return _builtin_call(float4x4, "INSTANCE_TO_WORLD_MATRIX", [self, instance_id])

    @callable
    def set_instance_transform(self: Accel, instance_id: int, mat: float4x4):
        _builtin_call("SET_INSTANCE_TRANSFORM", [self, instance_id, mat])

    @callable
    def set_instance_visibility(self: Accel, instance_id: int, vis: bool):
        _builtin_call("SET_INSTANCE_VISIBILITY", [self, instance_id, vis])


class Mesh:
    def __init__(self, vertices, triangles):
        assert vertices.dtype == float3
        assert triangles.dtype == int and triangles.size%3==0
        # TODO: support buffer of structs or arrays
        self.handle = get_global_device().impl().create_mesh(
            vertices.handle, 0, 16, vertices.size,
            triangles.handle, 0, triangles.size//3,
            lcapi.AccelUsageHint.FAST_TRACE)
        globalvars.stream.add(lcapi.MeshBuildCommand.create(
            self.handle, lcapi.AccelBuildRequest.PREFER_UPDATE,
            vertices.handle, 0, vertices.size,
            triangles.handle, 0, triangles.size//3))
