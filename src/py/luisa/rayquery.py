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
rayQueryAnyType = RayQueryAnyType()
def is_triangle(): ...
def is_procedural(): ...