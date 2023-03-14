#include <core/basic_types.h>
#include <array>
namespace luisa::compute {
struct AppData {
    float3 position;
    float3 normal;
    float4 tangent;
    float4 color;
    std::array<float2, 4> uv;
    uint32_t vertex_id;
    uint32_t instance_id;
};
}// namespace luisa::compute