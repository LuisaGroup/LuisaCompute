#pragma once

#include "metal_api.h"
#include <luisa/backends/ext/tex_compress_ext.h>

namespace luisa::compute::metal {

class MetalDevice;

class MetalTexCompressExt final : public TexCompressExt {

private:
    MetalDevice *_device;
    NS::SharedPtr<MTL::ComputePipelineState> _bc7_encode_try_mode_456;
    NS::SharedPtr<MTL::ComputePipelineState> _bc7_encode_try_mode_137;
    NS::SharedPtr<MTL::ComputePipelineState> _bc7_encode_try_mode_02;
    NS::SharedPtr<MTL::ComputePipelineState> _bc7_encode_encode_block;
    NS::SharedPtr<MTL::ComputePipelineState> _bc6h_encode_try_mode_g10;
    NS::SharedPtr<MTL::ComputePipelineState> _bc6h_encode_try_mode_le10;
    NS::SharedPtr<MTL::ComputePipelineState> _bc6h_encode_encode_block;

public:
    explicit MetalTexCompressExt(MetalDevice *device) noexcept;
    Result compress_bc6h(Stream &stream, const ImageView<float> &src, const BufferView<uint> &result) noexcept override;
    Result compress_bc7(Stream &stream, const ImageView<float> &src, const BufferView<uint> &result, float alpha_importance) noexcept override;
    Result check_builtin_shader() noexcept override;
};

}// namespace luisa::compute::metal
