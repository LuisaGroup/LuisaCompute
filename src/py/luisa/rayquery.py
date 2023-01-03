import lcapi
from .globalvars import get_global_device
from .types import to_lctype, basic_dtypes, dtype_of
from .types import vector_dtypes, matrix_dtypes, element_of, length_of
from functools import cache
from .func import func
from .builtin import _builtin_call, _bitwise_cast
from .struct import StructType
from .mathtypes import *
from .builtin import check_exact_signature
from .types import uint
from .struct import CustomType
from .hit import Hit, UHit
class RayQueryType:
    def __init__(self):
        self.luisa_type = lcapi.Type.custom("LC_RayQuery")

    def __eq__(self, other):
        return type(other) is RayQueryType and self.dtype == other.dtype

    def __hash__(self):
        return hash("LC_RayQuery") ^ 1641414112621983
    @func
    def proceed(self):
        return _builtin_call(bool, "RAY_QUERY_PROCEED", self)
    @func
    def is_triangle(self):
        return _builtin_call(bool, "RAY_QUERY_IS_CANDIDATE_TRIANGLE", self)
    @func
    def procedural_candidate(self):
        uhit = _builtin_call(UHit, "RAY_QUERY_PROCEDURAL_CANDIDATE_HIT", self)
        hit = Hit()
        hit.inst = _bitwise_cast(int, uhit.inst)
        hit.prim = _bitwise_cast(int, uhit.prim)
        hit.bary = uhit.bary
        hit.hit_type=_bitwise_cast(int, uhit.hit_type)
        hit.ray_t=uhit.ray_t
        return hit
    @func
    def triangle_candidate(self):
        uhit = _builtin_call(UHit, "RAY_QUERY_TRIANGLE_CANDIDATE_HIT", self)
        hit = Hit()
        hit.inst = _bitwise_cast(int, uhit.inst)
        hit.prim = _bitwise_cast(int, uhit.prim)
        hit.bary = uhit.bary
        hit.hit_type=_bitwise_cast(int, uhit.hit_type)
        hit.ray_t=uhit.ray_t
        return hit
    @func
    def get_commit_hit(self):
        uhit = _builtin_call(UHit, "RAY_QUERY_COMMITTED_HIT", self)
        hit = Hit()
        hit.inst = _bitwise_cast(int, uhit.inst)
        hit.prim = _bitwise_cast(int, uhit.prim)
        hit.bary = uhit.bary
        hit.hit_type=_bitwise_cast(int, uhit.hit_type)
        hit.ray_t=uhit.ray_t
        return hit
    @func
    def commit_triangle(self):
        _builtin_call("RAY_QUERY_COMMIT_TRIANGLE", self)
    @func
    def commit_procedural(self, distance:float):
        _builtin_call("RAY_QUERY_COMMIT_PROCEDURAL", self, distance)
rayQueryType = RayQueryType()
class RayQuery:
    def __init__(self):
        self.queryType = rayQueryType
        self.proceed = self.queryType.proceed
        self.is_triangle = self.queryType.is_triangle
        self.procedural_candidate = self.queryType.procedural_candidate
        self.triangle_candidate = self.queryType.triangle_candidate
        self.get_commit_hit = self.queryType.get_commit_hit
        self.commit_triangle = self.queryType.commit_triangle
        self.commit_procedural = self.queryType.commit_procedural
rayQuery = RayQuery()