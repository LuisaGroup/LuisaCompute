import lcapi
from .struct import StructType
from .func import func
from .types import uint
from .mathtypes import *

CommittedHit = StructType(inst=uint, prim=uint, bary=float2, hit_type=uint, ray_t=float)
TriangleHit = StructType(inst=uint, prim=uint, bary=float2, ray_t=float)
ProceduralHit = StructType(inst=uint, prim=uint)

@func
def _miss(self):
    return self.hit_type==0
@func
def _hit_triangle(self):
    return self.hit_type==1
@func
def _hit_procedural(self):
    return self.hit_type==2
CommittedHit.add_method(_miss, "miss")
CommittedHit.add_method(_hit_triangle, "hit_triangle")
CommittedHit.add_method(_hit_procedural, "hit_procedural")

@func
def _interpolate(self, a, b, c):
    return (1.0 - self.bary.x - self.bary.y) * a + self.bary.x * b + self.bary.y * c
CommittedHit.add_method(_interpolate, "interpolate")

@func
def _miss(self):
    return self.inst == 4294967295
@func
def _hitted(self):
    return self.inst != 4294967295
TriangleHit.add_method(_miss, "miss")
TriangleHit.add_method(_hitted, "hitted")
TriangleHit.add_method(_interpolate, "interpolate")
