import lcapi
from .globalvars import get_global_device
from .types import to_lctype, basic_dtypes, dtype_of
from .types import vector_dtypes, matrix_dtypes, element_of, length_of
from functools import cache
from .func import func
from .builtin import _builtin_call, bitwise_cast
from .struct import StructType
from .mathtypes import *
from .builtin import check_exact_signature
from .types import uint
from .struct import CustomType
from .hit import TriangleHit, CommittedHit, ProceduralHit
from .array import ArrayType

Ray = StructType(16, _origin=ArrayType(3,float), t_min=float, _dir=ArrayType(3,float), t_max=float)
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

class RayQueryAllType:
    def __init__(self):
        self.luisa_type = lcapi.Type.custom("LC_RayQueryAll")
    def __eq__(self, other):
        return type(other) is RayQueryAllType and self.dtype == other.dtype
    def __hash__(self):
        return hash("LC_RayQueryAll") ^ 1641414112621983
    @func
    def procedural_candidate(self):
        return _builtin_call(ProceduralHit, "RAY_QUERY_PROCEDURAL_CANDIDATE_HIT", self)
    @func
    def triangle_candidate(self):
        return _builtin_call(TriangleHit, "RAY_QUERY_TRIANGLE_CANDIDATE_HIT", self)
    @func
    def world_space_ray(self):
        return _builtin_call(Ray, "RAY_QUERY_WORLD_SPACE_RAY", self)
    @func
    def committed_hit(self):
        return _builtin_call(CommittedHit, "RAY_QUERY_COMMITTED_HIT", self)
    @func
    def terminate(self):
        _builtin_call("RAY_QUERY_TERMINATE", self)
    @func
    def commit_triangle(self):
        _builtin_call("RAY_QUERY_COMMIT_TRIANGLE", self)
    @func
    def commit_procedural(self, distance:float):
        _builtin_call("RAY_QUERY_COMMIT_PROCEDURAL", self, distance)
rayQueryAllType = RayQueryAllType()
class RayQueryAnyType:
    def __init__(self):
        self.luisa_type = lcapi.Type.custom("LC_RayQueryAny")
    def __eq__(self, other):
        return type(other) is RayQueryAnyType and self.dtype == other.dtype
    def __hash__(self):
        return hash("LC_RayQueryAny") ^ 2239219477752302592
    @func
    def procedural_candidate(self):
        return _builtin_call(ProceduralHit, "RAY_QUERY_PROCEDURAL_CANDIDATE_HIT", self)
    @func
    def triangle_candidate(self):
        return _builtin_call(TriangleHit, "RAY_QUERY_TRIANGLE_CANDIDATE_HIT", self)
    @func
    def committed_hit(self):
        return _builtin_call(CommittedHit, "RAY_QUERY_COMMITTED_HIT", self)
    @func
    def terminate(self):
        _builtin_call("RAY_QUERY_TERMINATE", self)
    @func
    def commit_triangle(self):
        _builtin_call("RAY_QUERY_COMMIT_TRIANGLE", self)
    @func
    def commit_procedural(self, distance:float):
        _builtin_call("RAY_QUERY_COMMIT_PROCEDURAL", self, distance)
    @func
    def world_space_ray(self):
        return _builtin_call(Ray, "RAY_QUERY_WORLD_SPACE_RAY", self)
rayQueryAnyType = RayQueryAnyType()
def is_triangle(): ...
def is_procedural(): ...