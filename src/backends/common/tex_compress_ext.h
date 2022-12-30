#pragma once

#include <runtime/device.h>

namespace luisa::compute {
class TexCompressExt : public DeviceExtension {

protected:
    virtual ~TexCompressExt() = default;

public:
    enum class Result : int8_t {
        NotImplemented = -1,
        Success = 0,
        Failed = 1
    };
    // TODO: astc
    virtual Result compress_bc6h(Stream &stream, Image<float> const &src, luisa::vector<std::byte> &result) noexcept { return Result::NotImplemented; }
    virtual Result compress_bc7(Stream &stream, Image<float> const &src, luisa::vector<std::byte> &result, float alpha_importance) noexcept { return Result::NotImplemented; }
    virtual Result check_builtin_shader() noexcept { return Result::NotImplemented; }
};
}// namespace luisa::compute