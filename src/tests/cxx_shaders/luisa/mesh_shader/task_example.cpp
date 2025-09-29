#include "attributes.hpp"
#include <luisa/std.hpp>
using namespace luisa::shader;
struct VertexIn {
	[[INSTANCE_ID]] uint instance_id;
	[[MESHLET_ID]] uint meshlet_id;
};
[[TASK_SHADER]] bool IsVisible(
	VertexIn vertex_in) {
    // Draw every mesh
	return true;
}