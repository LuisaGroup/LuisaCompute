#include "bindless_usage.h"
#include "texture_sampling.h"

namespace lc::spirv {

SpirvBindlessResourceUsage spirv_bindless_resource_usage(
    luisa::compute::xir::ResourceQueryOp op,
    luisa::compute::xir::BindlessResourceAccess access) noexcept {
    namespace xir = luisa::compute::xir;
    if (auto sample = spirv_texture_sample_op_info(op);
        sample.valid && !sample.direct) {
        return sample.is_2d ?
                   SpirvBindlessResourceUsage{.texture_2d = true} :
                   SpirvBindlessResourceUsage{.texture_3d = true};
    }
    switch (op) {
        case xir::ResourceQueryOp::BINDLESS_BUFFER_SIZE:
        case xir::ResourceQueryOp::BINDLESS_BYTE_BUFFER_SIZE:
            return {.buffer_metadata = !access.typed};
        case xir::ResourceQueryOp::BINDLESS_BUFFER_DEVICE_ADDRESS:
            return {.buffer_metadata = true};
        case xir::ResourceQueryOp::BINDLESS_TEXTURE2D_SIZE:
        case xir::ResourceQueryOp::BINDLESS_TEXTURE2D_SIZE_LEVEL:
            return {.texture_2d = true};
        case xir::ResourceQueryOp::BINDLESS_TEXTURE3D_SIZE:
        case xir::ResourceQueryOp::BINDLESS_TEXTURE3D_SIZE_LEVEL:
            return {.texture_3d = true};
        default: return {};
    }
}

SpirvBindlessResourceUsage spirv_bindless_resource_usage(
    luisa::compute::xir::ResourceReadOp op,
    luisa::compute::xir::BindlessResourceAccess access) noexcept {
    namespace xir = luisa::compute::xir;
    switch (op) {
        case xir::ResourceReadOp::BINDLESS_BUFFER_READ:
        case xir::ResourceReadOp::BINDLESS_BYTE_BUFFER_READ:
            return {
                .buffer_heap = true,
                .buffer_metadata = !access.typed};
        case xir::ResourceReadOp::BINDLESS_TEXTURE2D_READ:
        case xir::ResourceReadOp::BINDLESS_TEXTURE2D_READ_LEVEL:
            return {.texture_2d = true};
        case xir::ResourceReadOp::BINDLESS_TEXTURE3D_READ:
        case xir::ResourceReadOp::BINDLESS_TEXTURE3D_READ_LEVEL:
            return {.texture_3d = true};
        default: return {};
    }
}

SpirvBindlessResourceUsage spirv_bindless_resource_usage(
    luisa::compute::xir::ResourceWriteOp op,
    luisa::compute::xir::BindlessResourceAccess access) noexcept {
    namespace xir = luisa::compute::xir;
    switch (op) {
        case xir::ResourceWriteOp::BINDLESS_BUFFER_WRITE:
        case xir::ResourceWriteOp::BINDLESS_BYTE_BUFFER_WRITE:
            return {
                .buffer_heap = true,
                .buffer_metadata = !access.typed};
        default: return {};
    }
}

}// namespace lc::spirv
