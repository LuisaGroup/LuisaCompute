#include "attributes.hpp"
#include <luisa/std.hpp>
using namespace luisa::shader;
struct VertexOut {
	float4 position;
	float3 normal;
	float4 tangent;
	VertexOut(
		float4x4 transform,
		float4 position,
		float3 normal,
		float4 tangent)
		: position(transform * position),
		  normal((transform * float4(normal, 0.f)).xyz),
		  tangent(float4((transform * float4(tangent.xyz, 0.f)).xyz, tangent.w)) {}
};
struct MeshData {
	float4x4 transform;
	uint bindless_idx;
};
struct AppData {
	float4 position;
	float3 normal;
	float4 tangent;
};
struct VertexIn {
	[[INSTANCE_ID]] uint instance_id;
	// This can be unreliable due to backends' or task shaders' optimization, usually we don't need it.
	// [[MESHLET_ID]] uint meshlet_id;
	[[VERTEX_ID]] uint vertex_id;
};
[[VERTEX_SHADER]] VertexOut my_vertex_shader(
	Buffer<MeshData>& mesh_buffer,
	BindlessBuffer& heap,
	VertexIn vertex_in) {
	auto mesh_data = mesh_buffer.load(vertex_in.instance_id);
	auto appdata = heap.buffer_read<AppData>(mesh_data.bindless_idx, vertex_in.vertex_id);
	return VertexOut(
		mesh_data.transform,
		appdata.position,
		appdata.normal,
		appdata.tangent);
}