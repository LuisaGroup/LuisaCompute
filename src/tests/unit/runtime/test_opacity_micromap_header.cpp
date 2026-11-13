// Reproduce Vulkan's Linux include order without requiring X11 or a GPU.
// depth_format.h preserves X11's None macro before device.h includes micromaps.
#define None 0u
#include <luisa/runtime/depth_format.h>
#include <luisa/runtime/rtx/opacity_micromap.h>

#ifndef None
#error "The opacity micromap header must preserve the caller's None macro."
#endif
static_assert(None == 0u);
#undef None

using luisa::compute::OpacityMicromap;
static_assert(static_cast<uint32_t>(OpacityMicromap::IndexFormat::NoIndex) == 0u);
static_assert(OpacityMicromap::Option{}.index_format == OpacityMicromap::IndexFormat::NoIndex);
static_assert(OpacityMicromap::triangle_data_byte_size(OpacityMicromap::Format::TwoState, 2u) == 2u);

int main() {}
