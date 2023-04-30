from .dylibs import lcapi
from .globalvars import get_global_device
from .types import to_lctype, basic_dtypes, dtype_of
from .types import vector_dtypes, matrix_dtypes, element_of, length_of
from functools import cache
from .func import func
from .builtin import _builtin_call, bitwise_cast
from .struct import StructType
from .mathtypes import *
from .builtin import check_exact_signature
from .types import uint, BuiltinFuncBuilder
from .struct import CustomType
from .hit import TriangleHit, CommittedHit, ProceduralHit
from .array import ArrayType

Ray = StructType(16, _origin=ArrayType(3, float), t_min=float, _dir=ArrayType(3, float), t_max=float)


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

    @BuiltinFuncBuilder
    def procedural_candidate(*args):
        expr = lcapi.builder().call(to_lctype(ProceduralHit),
                                    lcapi.CallOp.RAY_QUERY_PROCEDURAL_CANDIDATE_HIT,
                                    [args[0].expr])
        return ProceduralHit, expr

    @BuiltinFuncBuilder
    def triangle_candidate(*args):
        expr = lcapi.builder().call(to_lctype(TriangleHit),
                                    lcapi.CallOp.RAY_QUERY_TRIANGLE_CANDIDATE_HIT,
                                    [args[0].expr])
        return TriangleHit, expr

    @BuiltinFuncBuilder
    def committed_hit(*args):
        expr = lcapi.builder().call(to_lctype(CommittedHit),
                                    lcapi.CallOp.RAY_QUERY_COMMITTED_HIT,
                                    [args[0].expr])
        return CommittedHit, expr

    @BuiltinFuncBuilder
    def terminate(*args):
        expr = lcapi.builder().call(lcapi.CallOp.RAY_QUERY_TERMINATE, [args[0].expr])
        return None, expr

    @BuiltinFuncBuilder
    def commit_triangle(*args):
        print(args)
        expr = lcapi.builder().call(lcapi.CallOp.RAY_QUERY_COMMIT_TRIANGLE, [args[0].expr])
        return None, expr

    @BuiltinFuncBuilder
    def commit_procedural(*args):
        expr = lcapi.builder().call(lcapi.CallOp.RAY_QUERY_COMMIT_PROCEDURAL, [args[0].expr, args[1].expr])
        return None, expr

    @BuiltinFuncBuilder
    def world_space_ray(*args):
        expr = lcapi.builder().call(to_lctype(Ray),
                                    lcapi.CallOp.RAY_QUERY_WORLD_SPACE_RAY,
                                    [args[0].expr])
        return Ray, expr


rayQueryAllType = RayQueryAllType()


class RayQueryAnyType:
    def __init__(self):
        self.luisa_type = lcapi.Type.custom("LC_RayQueryAny")

    def __eq__(self, other):
        return type(other) is RayQueryAnyType and self.dtype == other.dtype

    def __hash__(self):
        return hash("LC_RayQueryAny") ^ 2239219477752302592

    @BuiltinFuncBuilder
    def procedural_candidate(*args):
        expr = lcapi.builder().call(to_lctype(ProceduralHit),
                                    lcapi.CallOp.RAY_QUERY_PROCEDURAL_CANDIDATE_HIT,
                                    [args[0].expr])
        return ProceduralHit, expr

    @BuiltinFuncBuilder
    def triangle_candidate(*args):
        expr = lcapi.builder().call(to_lctype(TriangleHit),
                                    lcapi.CallOp.RAY_QUERY_TRIANGLE_CANDIDATE_HIT,
                                    [args[0].expr])
        return TriangleHit, expr

    @BuiltinFuncBuilder
    def committed_hit(*args):
        expr = lcapi.builder().call(to_lctype(CommittedHit),
                                    lcapi.CallOp.RAY_QUERY_COMMITTED_HIT,
                                    [args[0].expr])
        return CommittedHit, expr

    @BuiltinFuncBuilder
    def terminate(*args):
        expr = lcapi.builder().call(lcapi.CallOp.RAY_QUERY_TERMINATE, [args[0].expr])
        return None, expr

    @BuiltinFuncBuilder
    def commit_triangle(*args):
        print(args)
        expr = lcapi.builder().call(lcapi.CallOp.RAY_QUERY_COMMIT_TRIANGLE, [args[0].expr])
        return None, expr

    @BuiltinFuncBuilder
    def commit_procedural(*args):
        expr = lcapi.builder().call(lcapi.CallOp.RAY_QUERY_COMMIT_PROCEDURAL, [args[0].expr, args[1].expr])
        return None, expr

    @BuiltinFuncBuilder
    def world_space_ray(*args):
        expr = lcapi.builder().call(to_lctype(Ray),
                                    lcapi.CallOp.RAY_QUERY_WORLD_SPACE_RAY,
                                    [args[0].expr])
        return Ray, expr


rayQueryAnyType = RayQueryAnyType()


def is_triangle(): ...


def is_procedural(): ...
