import lcapi
from .struct import StructType
from .func import func
from .types import uint
from .mathtypes import *

# Hit
# hit_type: 0: miss, 1: triangle, 2: procedural primitive
Hit = StructType(inst=uint, prim=uint, bary=float2, hit_type=uint, ray_t=float)

@func
def miss(self):
    return self.hit_type==0
@func
def hit_triangle(self):
    return self.hit_type==1
@func
def hit_primitive(self):
    return self.hit_type==2
Hit.add_method(miss)
Hit.add_method(hit_triangle)
Hit.add_method(hit_primitive)

@func
def interpolate(self, a, b, c):
    return (1.0 - self.bary.x - self.bary.y) * a + self.bary.x * b + self.bary.y * c
Hit.add_method(interpolate)