#include <metal_stdlib>

using namespace metal;

struct alignas(16) Instance {
  array<float, 12> transform;
  uint options;
  uint mask;
  uint intersection_function_offset;
  uint mesh_index;
};

struct alignas(16) BuildRequest {
  uint index;
  uint flags;
  uint padding[2];
  float affine[12];
};

static_assert(sizeof(Instance) == 64u, "");
static_assert(sizeof(BuildRequest) == 64u, "");

[[kernel]]
void update_instance_buffer(
    device Instance *__restrict__ instances,
    device const BuildRequest *__restrict__ requests,
    constant uint &n,
    uint tid [[thread_position_in_grid]]) {
  if (tid < n) {
    auto r = requests[tid];
    instances[r.index].mesh_index = r.index;
    instances[r.index].options = 0x07u;
    instances[r.index].intersection_function_offset = 0u;
    constexpr auto update_flag_transform = 1u << 1u;
    constexpr auto update_flag_visibility_on = 1u << 2u;
    constexpr auto update_flag_visibility_off = 1u << 3u;
    constexpr auto update_flag_visibility = update_flag_visibility_on | update_flag_visibility_off;
    if (r.flags & update_flag_transform) {
      auto p = instances[r.index].transform.data();
      p[0] = r.affine[0];
      p[1] = r.affine[4];
      p[2] = r.affine[8];
      p[3] = r.affine[1];
      p[4] = r.affine[5];
      p[5] = r.affine[9];
      p[6] = r.affine[2];
      p[7] = r.affine[6];
      p[8] = r.affine[10];
      p[9] = r.affine[3];
      p[10] = r.affine[7];
      p[11] = r.affine[11];
    }
    if (r.flags & update_flag_visibility) {
      instances[r.index].mask = (r.flags & update_flag_visibility_on) ? ~0u : 0u;
    }
  }
}

struct alignas(16) BindlessItem {
  device const void *buffer;
  metal::uint sampler2d;
  metal::uint sampler3d;
  metal::texture2d<float> tex2d;
  metal::texture3d<float> tex3d;
};

[[kernel]]
void k(device const BindlessItem *array) {}

struct RasterData {
  float4 p [[position]];
  float2 uv;
};

[[vertex]]
RasterData v_simple(
  constant float4 *in [[buffer(0)]],
  uint vid [[vertex_id]]) {
  auto p = in[vid];
  return RasterData{p, saturate(p.xy * float2(.5f, -.5f) + .5f)};
}

[[fragment]]
float4 f_simple(
  RasterData in [[stage_in]],
  texture2d<float, access::sample> image [[texture(0)]]) {
  return float4(image.sample(sampler(filter::linear), in.uv).xyz, 1.f);
}
