#pragma once

#include <luisa/runtime/rhi/device_interface.h>

namespace luisa::compute {
template<typename T>
class BufferView;
class Stream;
template<typename T>
class Image;

class TexCompressExt : public DeviceExtension {

protected:
    ~TexCompressExt() noexcept = default;

public:
    static constexpr luisa::string_view name = "TexCompressExt";
    enum class Result : int8_t {
        NotImplemented = -1,
        Success = 0,
        Failed = 1
    };
    // TODO: astc
    virtual Result compress_bc6h(Stream &stream, Image<float> const &src, BufferView<uint> const &result) noexcept { return Result::NotImplemented; }
    virtual Result compress_bc7(Stream &stream, Image<float> const &src, BufferView<uint> const &result, float alpha_importance) noexcept { return Result::NotImplemented; }
    virtual Result check_builtin_shader() noexcept { return Result::NotImplemented; }
};

}// namespace luisa::compute
