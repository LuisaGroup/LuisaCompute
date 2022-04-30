import lcapi
from . import globalvars
from .globalvars import get_global_device
from .structtype import StructType
from .mathtypes import *

Ray = StructType(origin=float3, t_min=float, direction=float3, t_max=float)

Hit = StructType(inst=int, prim=int, bary=float2)
# Hit = StructType(inst=uint, prim=uint, bary=float2)


# Var<bool> miss(Expr<Hit> hit) noexcept {
#     return hit.inst == std::numeric_limits<uint>::max();
# }

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

class Mesh:
	def __init__(self, vertices, triangles):
		assert vertices.dtype == float3
		assert triangles.dtype == int and triangles.size%3==0
		self.handle = get_global_device().impl().create_mesh(
			vertices.handle, 0, 16, vertices.size,
			triangles.handle, 0, triangles.size//3,
			lcapi.AccelUsageHint.FAST_TRACE)
		globalvars.stream.add(lcapi.MeshBuildCommand.create(
			self.handle, lcapi.AccelBuildRequest.PREFER_UPDATE, vertices.handle, triangles.handle))
