import luisa
from .struct import StructType
from .array import ArrayType
from .types import *
appdata = StructType(position=float3, normal=float3, tangent=float4,color=float4, uv=ArrayType(4, float2), vertex_id=uint, instance_id=uint)
