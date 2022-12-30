#pragma once
#include <vstl/common.h>
#include <backends/common/tex_compress_ext.h>
using namespace luisa::compute;
namespace toolhub::directx {
class Device;
class DxTexCompressExt final : public TexCompressExt, public vstd::IOperatorNewBase {
public:
    static constexpr size_t BLOCK_SIZE = 16;
    Device *device;
    DxTexCompressExt(Device *device);
    ~DxTexCompressExt();
    Result compress_bc6h(Stream &stream, Image<float> const &src, vstd::vector<std::byte> &result) noexcept override;
    Result compress_bc7(Stream &stream, Image<float> const &src, vstd::vector<std::byte> &result, float alphaImportance) noexcept override;
    Result check_builtin_shader() noexcept override;
};
}// namespace toolhub::directx 