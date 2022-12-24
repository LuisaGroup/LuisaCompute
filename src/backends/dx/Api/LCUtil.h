#pragma once
#include <vstl/common.h>
#include <runtime/util.h>
using namespace luisa::compute;
namespace toolhub::directx {
class Device;
class LCUtil final : public IUtil, public vstd::IOperatorNewBase {
public:
    static constexpr size_t BLOCK_SIZE = 16;
    Device *device;
    LCUtil(Device *device);
    ~LCUtil();
    Result compress_bc6h(Stream &stream, Image<float> const &src, vstd::vector<std::byte> &result) noexcept override;
    Result compress_bc7(Stream &stream, Image<float> const &src, vstd::vector<std::byte> &result, float alphaImportance) noexcept override;
    Result check_builtin_shader() noexcept override;
};
}// namespace toolhub::directx